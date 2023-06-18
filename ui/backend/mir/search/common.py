import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Literal

import numpy as np
import pandas as pd

from mir.clustering.clustering import cluster_kmeans
from mir.clustering.embeddings import load_docs_embedding
from mir.clustering.preprocess import preprocess_text
from mir.search.corpus.phase1 import Corpus as Phase1Corpus
from mir.search.corpus.phase2 import Corpus as Phase2Corpus
from mir.search.corpus.phase3 import Corpus as Phase3Corpus
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
class Phase1:
    corpus: Phase1Corpus
    trie: Phase1TrieNode
    bigram_index: Dict[str, Dict[str, int]]


@dataclass
class Phase2:
    corpus: Phase2Corpus
    trie: Phase2TrieNode
    bigram_index: Dict[str, Dict[str, int]]
    kmeans_dict: dict


@dataclass
class Phase3:
    corpus: Phase3Corpus
    trie: Phase3TrieNode
    bigram_index: Dict[str, Dict[str, int]]


@dataclass
class SimilarPapers:
    docs_embedding: np.ndarray
    data: pd.DataFrame


class Phase3Professors(Enum):
    Kasaei = 'Kasaei'
    Rabiee = 'Rabiee'
    Rohban = 'Rohban'
    Sharifi = 'Sharifi'
    Soleymani = 'Soleymani'


@pickle_cache
def load_corpus(dataset: Literal['ai-bio', 'hardware-system', 'arxiv', 'ss']):
    if dataset in ('ai-bio', 'hardware-system'):
        corpus = Phase1Corpus(dataset, stop_topk=20)
    elif dataset == 'arxiv':
        corpus = Phase2Corpus('../data/arxiv/arxiv_data_210930-054931.csv', stop_topk=20)
    elif dataset == 'ss':
        papers_by_id = {}
        papers_by_professor = {}
        for professor in Phase3Professors:
            with open(f'../results/crawled_paper_{professor.value}.json') as f:
                papers_by_professor[professor.value] = [p for p in json.load(f) if np.random.rand() < 0.5]
                for paper in papers_by_professor[professor.value]:
                    papers_by_id[paper['id']] = paper
        all_papers = list(papers_by_id.values())
        corpus = Phase3Corpus(papers_by_professor, all_papers, stop_topk=20)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    _ = corpus.non_stop_documents
    return corpus


def load_phase1(dataset: Literal['ai-bio', 'hardware-system']):
    corpus = load_corpus(dataset)
    trie = phase1_construct_positional_indexes(dataset, corpus)
    bigram_index = phase1_create_bigram_index(dataset, corpus)
    return Phase1(corpus, trie, bigram_index)


@pickle_cache(args_for_hash=[])
def process_arxiv_data(_data: pd.DataFrame) -> pd.Series:
    return _data['titles'].str.cat(_data['abstracts'], sep=' ') \
        .apply(preprocess_text).str.join(' ')


def load_phase2():
    dataset = 'arxiv'
    corpus = load_corpus(dataset)
    trie = phase2_construct_positional_indexes(dataset, corpus)
    bigram_index = phase2_create_bigram_index(dataset, corpus)
    docs_embedding = load_docs_embedding('../drive/MyDrive/arxiv-sbert-embeddings.npy')[corpus.random_indices]
    preprocessed_documents = process_arxiv_data(corpus.data)
    kmeans_dict = cluster_kmeans(docs_embedding, preprocessed_documents, 3)
    return Phase2(corpus, trie, bigram_index, kmeans_dict)


def load_phase3():
    dataset = 'ss'
    corpus = load_corpus(dataset)
    trie = phase3_construct_positional_indexes(dataset, corpus)
    bigram_index = phase3_create_bigram_index(dataset, corpus)
    return Phase3(corpus, trie, bigram_index)


def load_arxiv_data() -> pd.DataFrame:
    df = pd.read_csv('../data/arxiv/arxiv_data_210930-054931.csv').fillna('')
    df['category'] = df['terms'].apply(lambda l: eval(l)[0])
    df = df.drop_duplicates(['titles']).drop_duplicates(['abstracts']).reset_index(drop=True)
    df['paper_id'] = df.index.astype(str)
    return df


def load_similar_papers():
    docs_embedding = load_docs_embedding('../drive/MyDrive/arxiv-sbert-embeddings.npy')
    data = load_arxiv_data()
    return SimilarPapers(docs_embedding, data)
