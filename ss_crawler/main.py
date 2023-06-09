import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import logging
import math
import os
import re
import time
import traceback
from typing import List

from redis_decorators import RedisCaching
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

MAX_PAPERS = int(os.getenv('MAX_PAPERS', 1000))
MIN_WAIT_TIME = float(os.getenv('MIN_WAIT_TIME', 10))
MAX_WAIT_TIME = float(os.getenv('MAX_WAIT_TIME', 60))
cache = RedisCaching('redis://cache:6379')


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    publication_year: int
    authors: List[str]
    related_topics: List[str]
    citation_count: int
    reference_count: int
    references: List[str]

    @staticmethod
    def serialize(func):
        def wrapper(*args, **kwargs):
            return json.dumps(asdict(func(*args, **kwargs)))
        return wrapper

    @staticmethod
    def deserialize(func):
        def wrapper(*args, **kwargs):
            return Paper(**json.loads(func(*args, **kwargs)))
        return wrapper


class PageNotFoundError(Exception):
    pass


failure_counts_by_wait_power = defaultdict(lambda: defaultdict(int))
total_tries = defaultdict(int)
failure_rate_threshold = 0.2
max_wait_power = int(math.log2(MAX_WAIT_TIME / MIN_WAIT_TIME))


def calculate_wait_power(func_name):
    n = total_tries[func_name]
    if n == 0:
        return 0

    acceptable_powers = [
        power
        for power, failure_count in failure_counts_by_wait_power[func_name].items()
        if failure_count / n < failure_rate_threshold
    ]

    if not acceptable_powers:
        return 0

    return min(max_wait_power, min(acceptable_powers))


def retry(func=None):
    # exponential backoff

    def decorator(func):
        def wrapper(*args, **kwargs):
            wait_power = calculate_wait_power(func.__name__)

            i = 1
            while True:
                try:
                    total_tries[func.__name__] += 1
                    result = func(*args, **kwargs)
                    time.sleep(MIN_WAIT_TIME * 2 ** calculate_wait_power(func.__name__))
                    return result
                except (KeyboardInterrupt, SystemExit, PageNotFoundError):
                    raise
                except:
                    print(f'Failed to call {func.__name__}({args}, {kwargs})')
                    traceback.print_exc()
                    failure_counts_by_wait_power[func.__name__][wait_power] += 1
                    if wait_power < max_wait_power:
                        wait_power += 1
                    wait_time = MIN_WAIT_TIME * 2 ** wait_power
                    print(f'Retry {i} after {wait_time:.2f}s wait')
                    time.sleep(wait_time)
                    i += 1
        return wrapper

    if func:
        return decorator(func)
    return decorator


def id_from_url(url: str) -> str:
    return url.split('/')[-1]


def url_from_id(id_: str) -> str:
    return f'https://www.semanticscholar.org/paper/{id_}'


def print_call(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'{func.__name__}({args}, {kwargs}) -> {result}')
        return result
    return wrapper


class SemanticScholarCrawler:
    citation_count_pattern = re.compile(r'(?P<count>[\d,]+) Citations')
    reference_count_pattern = re.compile(r'(?P<count>[\d,]+) References')

    def __init__(self, driver: webdriver.Chrome, seed_urls: List[str]):
        self.driver = driver
        self.queue = [id_from_url(url.strip()) for url in seed_urls]
        self.papers: List[Paper] = []
        self.stored_ids = set()
        self.crawled_count = 0

    @property
    def has_next(self) -> bool:
        return bool(self.queue) and len(self.papers) < MAX_PAPERS

    def pop_new_id(self) -> str:
        while (id_ := self.queue.pop(0)) in self.stored_ids:
            pass
        return id_

    def retrieve_paper_properly(self, id_: str):
        paper = self.crawl_paper(id_)
        self.stored_ids.add(id_)
        self.papers.append(paper)
        self.queue.extend(paper.references)

    def get_next(self):
        while self.has_next:
            id_ = self.pop_new_id()
            try:
                self.retrieve_paper_properly(id_)
                break
            except PageNotFoundError:
                print(f'Page not found: {id_}. Skipping...')
                continue

    def raise_if_page_not_found(self):
        if (status_element := next(iter(self.driver.find_elements(By.ID, 'error-status')), None))\
                and status_element.get_attribute('value') == '404':
            raise PageNotFoundError()

    @Paper.deserialize
    @cache.cache_string(get_cache_key=lambda self, id_: f'ss-paper:{id_}')
    @Paper.serialize
    @retry
    def crawl_paper(self, id_: str) -> Paper:
        url = url_from_id(id_)
        self.driver.get(url)
        self.raise_if_page_not_found()
        paper = Paper(
            id=id_,
            title=self.current_title,
            abstract=self.current_abstract,
            publication_year=self.current_publication_year,
            authors=self.current_authors,
            related_topics=self.current_related_topics,
            citation_count=self.current_citation_count,
            reference_count=self.current_reference_count,
            references=self.current_references)
        self.crawled_count += 1
        return paper

    # @print_call
    def get_single_meta(self, name: str) -> str:
        return self.driver.find_element(By.CSS_SELECTOR, f'meta[name="{name}"]').get_attribute('content')

    # @print_call
    def get_multiple_meta(self, name: str) -> List[str]:
        return [a.get_attribute('content') for a in self.driver.find_elements(By.CSS_SELECTOR, f'meta[name="{name}"]')]

    @property
    def current_title(self) -> str:
        return self.get_single_meta('citation_title')

    @property
    def current_abstract(self) -> str:
        return self.get_single_meta('description')

    @property
    def current_publication_year(self) -> int:
        try:
            return int(self.get_single_meta('citation_publication_date'))
        except NoSuchElementException:
            return -1

    @property
    def current_authors(self) -> List[str]:
        return self.get_multiple_meta('citation_author')

    @property
    def current_related_topics(self) -> List[str]:
        return next((li.text.split(', ')
                     for li in self.driver.find_elements(By.CSS_SELECTOR, 'li.paper-meta-item')
                     if not li.find_elements(By.CSS_SELECTOR, 'span')), [])

    @property
    def current_citation_count(self) -> int:
        return int(next((mo.group('count').replace(',', '')
                         for x in self.driver.find_elements(By.CSS_SELECTOR, 'span.paper-nav__nav-label')
                         if (mo := self.citation_count_pattern.match(x.text))), -1))

    @property
    def current_reference_count(self) -> int:
        return int(next((mo.group('count').replace(',', '')
                        for x in self.driver.find_elements(By.CSS_SELECTOR, 'span.paper-nav__nav-label')
                        if (mo := self.reference_count_pattern.match(x.text))), -1))

    @property
    def current_references(self) -> List[str]:
        references_section = next(iter(self.driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="reference"]')),
                                  None)
        if references_section:
            return [id_
                    for div in references_section
                    .find_element(By.CSS_SELECTOR, 'div.citation-list__citations')
                    .find_elements(By.XPATH, './/div[@data-paper-id]')
                    if (id_ := div.get_attribute('data-paper-id'))]
        return []


def main(args: argparse.Namespace):
    print(args)
    with open(args.input, 'r') as f:
        urls = f.readlines()

    chrome_options = webdriver.ChromeOptions()
    prefs = {'download.default_directory': os.getcwd()}
    chrome_options.add_experimental_option('prefs', prefs)
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--headless')

    caps = DesiredCapabilities().CHROME
    caps["pageLoadStrategy"] = "eager"  # interactive

    service = Service(ChromeDriverManager(print_first_line=False, log_level=logging.INFO).install())
    driver = webdriver.Chrome(service=service,
                              desired_capabilities=caps,
                              options=chrome_options)

    crawler = SemanticScholarCrawler(driver, urls)

    with tqdm(total=MAX_PAPERS) as pbar:
        while crawler.has_next:
            crawler.get_next()
            pbar.update(1)
            pbar.set_postfix(dict(
                queue_size=len(crawler.queue),
                crawled_count=crawler.crawled_count,
            ))

    with open(args.output, 'w') as f:
        json.dump([asdict(paper) for paper in crawler.papers], f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    args = parser.parse_args()
    main(args)
