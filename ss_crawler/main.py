import argparse
import os
import re
from dataclasses import dataclass, asdict
import json
import logging
import time
from typing import List

from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from tqdm import tqdm

MAX_PAPERS = int(os.getenv('MAX_PAPERS', 1000))
WAIT_TIME = float(os.getenv('WAIT_TIME', 10))


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


def retry(func=None, max_retries: int = 5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            e = None
            for i in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f'Failed to call {func.__name__}({args}, {kwargs})')
                    print(e)
                    if i < max_retries:
                        print(f'Retry {i + 1}/{max_retries}')
                    time.sleep(WAIT_TIME)
            raise e
        return wrapper

    if func:
        return decorator(func)
    return decorator


@retry
def crawl_paper(id_: str) -> Paper:
    url = url_from_id(id_)
    response = session.get(url, headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
        'sec-ch-ua': '"Chromium";v="94", "Google Chrome";v="94", ";Not A Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
    })
    if response.status_code != 200:
        raise Exception(f'Failed to crawl {url}, status code: {response.status_code}')
    soup = bs4.BeautifulSoup(response.text, 'html.parser')
    return Paper(
        id=id_,
        title=soup.find('meta', {'name': 'citation_title'})['content'],
        abstract=soup.find('meta', {'name': 'description'})['content'],
        publication_year=int(soup.find('meta', {'name': 'citation_publication_date'})['content']),
        authors=[a.text for a in soup.find_all('meta', {'name': 'citation_author'})],
        related_topics=next((li.text.split(', ')
                             for li in soup.find_all('li', {'class': 'paper-meta-item'})
                             if li.find('span') is None), []),
        citation_count=int(soup.find_all('span', {'class': 'paper-nav__nav-label'})[0].text.split()[0]),
        reference_count=int(soup.find_all('span', {'class': 'paper-nav__nav-label'})[1].text.split()[0]),
        references=[div['data-paper-id']
                    for div in soup.find('div', {'data-test-id': 'reference'})
                    .find('div', {'class': 'citation-list__citations'})])


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
    citation_count_pattern = re.compile(r'(?P<count>\d+) Citations')
    reference_count_pattern = re.compile(r'(?P<count>\d+) References')

    def __init__(self, driver: webdriver.Chrome, seed_urls: List[str]):
        self.driver = driver
        self.queue = [id_from_url(url.strip()) for url in seed_urls]
        self.papers: List[Paper] = []

    @property
    def has_next(self) -> bool:
        return bool(self.queue) and len(self.papers) < MAX_PAPERS

    def get_next(self):
        id_ = self.queue.pop(0)
        paper = self.crawl_paper(id_)
        print(f'crawled {paper}')
        self.papers.append(paper)
        self.queue.extend(paper.references)

    @retry
    def crawl_paper(self, id_: str) -> Paper:
        url = url_from_id(id_)
        self.driver.get(url)
        return Paper(
            id=id_,
            title=self.current_title,
            abstract=self.current_abstract,
            publication_year=self.current_publication_year,
            authors=self.current_authors,
            related_topics=self.current_related_topics,
            citation_count=self.current_citation_count,
            reference_count=self.current_reference_count,
            references=self.current_references)

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
        return int(self.get_single_meta('citation_publication_date'))

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
        return int(next(mo.group('count')
                        for x in self.driver.find_elements(By.CSS_SELECTOR, 'span.paper-nav__nav-label')
                        if (mo := self.citation_count_pattern.match(x.text))))

    @property
    def current_reference_count(self) -> int:
        return int(next(mo.group('count')
                        for x in self.driver.find_elements(By.CSS_SELECTOR, 'span.paper-nav__nav-label')
                        if (mo := self.reference_count_pattern.match(x.text))))

    @property
    def current_references(self) -> List[str]:
        return [div.get_attribute('data-paper-id')
                for div in self.driver.find_element(By.CSS_SELECTOR, 'div[data-test-id="reference"]')
                .find_element(By.CSS_SELECTOR, 'div.citation-list__citations')
                .find_elements(By.XPATH, './/div[@data-paper-id]')]


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
            ))
            time.sleep(WAIT_TIME)

    with open(args.output, 'w') as f:
        json.dump([asdict(paper) for paper in crawler.papers], f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file path')
    args = parser.parse_args()
    main(args)
