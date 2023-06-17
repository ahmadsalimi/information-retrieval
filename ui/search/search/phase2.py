from dataclasses import dataclass
from typing import Dict, Literal, List, Iterator

from search.bigram_index.phase2 import correct_text
from search.corpus.phase2 import Corpus
from search.preprocess import clean_data
from search.search.common import tf_idf, okapi25, highlight_text
from search.trie.phase2 import TrieNode


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str
    abstract: str
    category: str
    cluster: str


def search(corpus: Corpus, trie: TrieNode, bigram_index: Dict[str, Dict[str, int]], query: str, max_result_count: int,
           method: str = 'ltn-lnn', weight: float = 0.5, highlight: bool = False, category: str = 'all',
           kmeans_dict: dict = None) -> Iterator:
    """
        Finds relevant documents to query

        Parameters
        ---------------------------------------------------------------------------------------------------
        corpus: Corpus
            The corpus
        trie: TrieNode
            The trie
        bigram_index: Dict[str, Dict[str, int]]
            The bigram index
        query: str
            The query string
        max_result_count: int
            Return top 'max_result_count' docs which have the highest scores.
            notice that if max_result_count = -1, then you have to return all docs
        method: 'ltn-lnn' or 'ltc-lnc' or 'okapi25'
        weight: float
            weight of abstract score
        highlight: bool
            If True, highlight the query tokens in search results
        print_result: bool
            If True, print the results in a readable format
        category: str
            If 'all', search in all documents, else search in documents of the given category

        Returns
        ----------------------------------------------------------------------------------------------------
        list
            Retrieved documents with snippet
    """
    corrected_query = correct_text(bigram_index, query)
    yield corrected_query
    query_tokens = [token.processed
                    for token in clean_data(corrected_query)
                    if token.processed not in corpus.stop_tokens]
    token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]] = {
        token: trie.search(token) or {}
        for token in query_tokens
    }
    token_search_results = {
        token: search_results if category == 'all' else {
            doc_id: doc
            for doc_id, doc in search_results.items()
            if doc_id in corpus.document_ids_by_category[category]
        }
        for token, search_results in token_search_results.items()
    }
    if '-' in method:
        title_doc_scores = tf_idf(corpus, 'title', token_search_results, query_tokens, method)
        abstract_doc_scores = tf_idf(corpus, 'abstract', token_search_results, query_tokens, method)
    elif method == 'okapi25':
        title_doc_scores = okapi25(corpus, 'title', token_search_results, query_tokens)
        abstract_doc_scores = okapi25(corpus, 'abstract', token_search_results, query_tokens)
    else:
        raise ValueError(f'Expected the method to be one of \'ltn-lnn\', \'ltc-lnc\', or \'okapi25\', bot got {method}')
    doc_scores = {
        doc_id: weight * abstract_doc_scores.get(doc_id, 0.0) + (1 - weight) * title_doc_scores.get(doc_id, 0.0)
        for doc_id in set(title_doc_scores.keys()) | set(abstract_doc_scores.keys())
    }
    if highlight:
        doc_highlights = {
            doc_id: {
                'title': sum([
                    list(token_search_results[token][doc_id]['title'])
                    for token in query_tokens
                    if token in token_search_results and doc_id in token_search_results[token]
                ], []),
                'abstract': sum([
                    list(token_search_results[token][doc_id]['abstract'])
                    for token in query_tokens
                    if token in token_search_results and doc_id in token_search_results[token]
                ], [])
            }
            for doc_id in doc_scores
        }
    else:
        doc_highlights = {
            doc_id: {
                'title': [],
                'abstract': [],
            }
            for doc_id in doc_scores
        }
    if kmeans_dict is not None:
        cluster_topics = {
            doc_id: kmeans_dict['main_topics'][kmeans_dict['cluster_indices'][int(doc_id)]]
            for doc_id in doc_scores
        }
    else:
        cluster_topics = None
    documents = [
        SearchResult(
            doc_id=doc_id,
            score=score,
            title=highlight_text(corpus.data[corpus.data['paper_id'] == doc_id]['titles'].item(),
                                 doc_highlights[doc_id]['title']),
            abstract=highlight_text(corpus.data[corpus.data['paper_id'] == doc_id]['abstracts'].item(),
                                    doc_highlights[doc_id]['abstract']),
            category=corpus.data[corpus.data['paper_id'] == doc_id]['category'].item(),
            cluster=cluster_topics[doc_id] if cluster_topics is not None else None,
        )
        for i, (doc_id, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
        if i < max_result_count or max_result_count == -1
    ]
    yield documents
