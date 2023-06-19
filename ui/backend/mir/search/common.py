import contextlib
import json
import threading
from concurrent import futures
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal, Optional, Any, Callable, Tuple, Iterable

import nltk
import numpy as np
import pandas as pd

from mir.clustering.clustering import cluster_kmeans
from mir.clustering.embeddings import load_docs_embedding
from mir.clustering.preprocess import preprocess_text
from mir.search.corpus.phase1 import Corpus as Phase1Corpus
from mir.search.corpus.phase2 import Corpus as Phase2Corpus
from mir.search.corpus.phase3 import Corpus as Phase3Corpus
from mir.search.preprocess import load_nlp
from mir.search.trie.phase1 import TrieNode as Phase1TrieNode,\
    construct_positional_indexes as phase1_construct_positional_indexes
from mir.search.trie.phase2 import TrieNode as Phase2TrieNode,\
    construct_positional_indexes as phase2_construct_positional_indexes
from mir.search.trie.phase3 import TrieNode as Phase3TrieNode,\
    construct_positional_indexes as phase3_construct_positional_indexes
from mir.search.bigram_index.phase1 import create_bigram_index as phase1_create_bigram_index
from mir.search.bigram_index.phase2 import create_bigram_index as phase2_create_bigram_index
from mir.search.bigram_index.phase3 import create_bigram_index as phase3_create_bigram_index
from mir.util import pickle_cache


@dataclass
class LazyLoadedDataClass:
    @property
    def is_totally_loaded(self) -> bool:
        return all(value is not None for value in self.__dict__.values())


@dataclass
class Phase1(LazyLoadedDataClass):
    corpus: Optional[Phase1Corpus] = None
    trie: Optional[Phase1TrieNode] = None
    bigram_index: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class Phase2(LazyLoadedDataClass):
    corpus: Optional[Phase2Corpus] = None
    trie: Optional[Phase2TrieNode] = None
    bigram_index: Optional[Dict[str, Dict[str, int]]] = None
    kmeans_dict: Optional[dict] = None


@dataclass
class Phase3(LazyLoadedDataClass):
    corpus: Optional[Phase3Corpus] = None
    trie: Optional[Phase3TrieNode] = None
    bigram_index: Optional[Dict[str, Dict[str, int]]] = None


@dataclass
class SimilarPapers(LazyLoadedDataClass):
    docs_embedding: Optional[np.ndarray] = None
    data: Optional[pd.DataFrame] = None


class Phase3Professors(Enum):
    Kasaei = 'Kasaei'
    Rabiee = 'Rabiee'
    Rohban = 'Rohban'
    Sharifi = 'Sharifi'
    Soleymani = 'Soleymani'


@pickle_cache
def load_corpus(dataset: Literal['ai-bio', 'hardware-system', 'arxiv', 'ss']):
    if dataset in ('ai-bio', 'hardware-system'):
        return Phase1Corpus(dataset, stop_topk=20)
    elif dataset == 'arxiv':
        return Phase2Corpus('../data/arxiv/arxiv_data_210930-054931.csv', stop_topk=20)
    elif dataset == 'ss':
        papers_by_id = {}
        papers_by_professor = {}
        for professor in Phase3Professors:
            with open(f'../results/crawled_paper_{professor.value}.json') as f:
                papers_by_professor[professor.value] = [p for p in json.load(f) if np.random.rand() < 0.5]
                for paper in papers_by_professor[professor.value]:
                    papers_by_id[paper['id']] = paper
        all_papers = list(papers_by_id.values())
        return Phase3Corpus(papers_by_professor, all_papers, stop_topk=20)
    else:
        raise ValueError(f'Unknown dataset {dataset}')


class DependentRunner:
    set_lock = threading.Lock()
    current_context_default_dependencies: Iterable['DependentRunner'] = []

    def __init__(self, loader: Callable[[], Any],
                 setter: Optional[Callable[[Any], None]] = None,
                 *dependencies: 'DependentRunner'):
        self.loader = loader
        self.setter = setter
        self.dependencies = list(self.current_context_default_dependencies) + list(dependencies)
        self.event = threading.Event()

    def __call__(self):
        for dependency in self.dependencies:
            dependency.event.wait()
        result = self.loader()
        if self.setter is not None:
            with self.set_lock:
                self.setter(result)
        self.event.set()

    def submit_to(self, executor: futures.ThreadPoolExecutor) -> 'DependentRunner':
        executor.submit(self)
        return self

    @classmethod
    def attr_setter(cls, obj: Any, attr: str,
                    loader: Callable[[], Any],
                    *dependencies: 'DependentRunner') -> 'DependentRunner':
        return cls(loader, lambda result: setattr(obj, attr, result), *dependencies)

    @classmethod
    def dict_setter(cls, d: Dict[str, Any], key: str,
                    loader: Callable[[], Any],
                    *dependencies: 'DependentRunner') -> 'DependentRunner':
        return cls(loader, lambda result: d.__setitem__(key, result), *dependencies)

    @classmethod
    @contextlib.contextmanager
    def use_default_dependencies(cls, *dependencies: 'DependentRunner'):
        old_default_dependencies = cls.current_context_default_dependencies
        cls.current_context_default_dependencies = dependencies
        try:
            yield
        finally:
            cls.current_context_default_dependencies = old_default_dependencies


def load_languages(executor: futures.ThreadPoolExecutor) -> Tuple[DependentRunner, ...]:
    stopwords = DependentRunner(lambda: nltk.download('stopwords')).submit_to(executor)
    punkt = DependentRunner(lambda: nltk.download('punkt')).submit_to(executor)
    nlp = DependentRunner(load_nlp).submit_to(executor)
    return stopwords, punkt, nlp


def load_phase1(executor: futures.ThreadPoolExecutor,
                dataset: Literal['ai-bio', 'hardware-system']) -> Phase1:
    phase1 = Phase1()
    corpus = DependentRunner.attr_setter(phase1, 'corpus',
                                         lambda: load_corpus(dataset))\
        .submit_to(executor)
    DependentRunner.attr_setter(phase1, 'trie',
                                lambda: phase1_construct_positional_indexes(dataset, phase1.corpus), corpus)\
        .submit_to(executor)
    DependentRunner.attr_setter(phase1, 'bigram_index',
                                lambda: phase1_create_bigram_index(dataset, phase1.corpus), corpus)\
        .submit_to(executor)
    return phase1


@pickle_cache(args_for_hash=[])
def process_arxiv_data(_data: pd.DataFrame) -> pd.Series:
    return _data['titles'].str.cat(_data['abstracts'], sep=' ') \
        .apply(preprocess_text).str.join(' ')


def load_phase2(executor: futures.ThreadPoolExecutor) -> Phase2:
    dataset = 'arxiv'
    phase2 = Phase2()
    intermediate_objects = {}
    corpus = DependentRunner.attr_setter(phase2, 'corpus',
                                         lambda: load_corpus(dataset))\
        .submit_to(executor)
    DependentRunner.attr_setter(phase2, 'trie',
                                lambda: phase2_construct_positional_indexes(dataset, phase2.corpus), corpus)\
        .submit_to(executor)
    DependentRunner.attr_setter(phase2, 'bigram_index',
                                lambda: phase2_create_bigram_index(dataset, phase2.corpus), corpus)\
        .submit_to(executor)
    docs_embedding = DependentRunner.dict_setter(intermediate_objects, 'docs_embedding',
                                                 lambda: load_docs_embedding(
                                                     '../drive/MyDrive/arxiv-sbert-embeddings.npy'))\
        .submit_to(executor)
    preprocessed_documents = DependentRunner.dict_setter(intermediate_objects, 'preprocessed_documents',
                                                         lambda: process_arxiv_data(phase2.corpus.data), corpus) \
        .submit_to(executor)
    DependentRunner.attr_setter(phase2, 'kmeans_dict',
                                lambda: cluster_kmeans(
                                    intermediate_objects['docs_embedding'][phase2.corpus.random_indices],
                                    intermediate_objects['preprocessed_documents'], 3),
                                corpus, docs_embedding, preprocessed_documents) \
        .submit_to(executor)
    return phase2


def load_phase3(executor: futures.ThreadPoolExecutor) -> Phase3:
    dataset = 'ss'
    phase3 = Phase3()
    corpus = DependentRunner.attr_setter(phase3, 'corpus',
                                         lambda: load_corpus(dataset))\
        .submit_to(executor)
    DependentRunner.attr_setter(phase3, 'trie',
                                lambda: phase3_construct_positional_indexes(dataset, phase3.corpus), corpus)\
        .submit_to(executor)
    DependentRunner.attr_setter(phase3, 'bigram_index',
                                lambda: phase3_create_bigram_index(dataset, phase3.corpus), corpus)\
        .submit_to(executor)
    return phase3


def load_arxiv_data() -> pd.DataFrame:
    df = pd.read_csv('../data/arxiv/arxiv_data_210930-054931.csv').fillna('')
    df['category'] = df['terms'].apply(lambda l: eval(l)[0])
    df = df.drop_duplicates(['titles']).drop_duplicates(['abstracts']).reset_index(drop=True)
    df['paper_id'] = df.index.astype(str)
    return df


def load_similar_papers(executor: futures.ThreadPoolExecutor) -> SimilarPapers:
    similar_papers = SimilarPapers()
    DependentRunner.attr_setter(similar_papers, 'docs_embedding',
                                lambda: load_docs_embedding('../drive/MyDrive/arxiv-sbert-embeddings.npy'))\
        .submit_to(executor)
    DependentRunner.attr_setter(similar_papers, 'data', load_arxiv_data)\
        .submit_to(executor)
    return similar_papers
