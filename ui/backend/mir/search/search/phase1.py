from dataclasses import dataclass
from typing import Dict, Literal, List, Iterator

from mir.search.bigram_index.phase1 import correct_text
from mir.search.corpus.phase1 import Corpus
from mir.search.preprocess import clean_data, nlp
from mir.search.search.common import tf_idf, okapi25, highlight_text
from mir.search.trie.phase1 import TrieNode


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str
    abstract: str


def search(corpus: Corpus, trie: TrieNode, bigram_index: Dict[str, Dict[str, int]], query: str, max_result_count: int,
           highlight: bool = False,
           method: str = 'ltn-lnn', weight: float = 0.5) -> Iterator:
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

        Returns
        ----------------------------------------------------------------------------------------------------
        list
            Retrieved documents with snippet
    """
    corrected_query = correct_text(bigram_index, query)
    yield corrected_query
    query_tokens = [token.processed
                    for token in nlp(corrected_query)
                    if token.processed not in corpus.stop_tokens]
    token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]] = {
        token: trie.search(token) or {}
        for token in query_tokens
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
    documents = [
        SearchResult(
            doc_id=doc_id,
            score=score,
            title=highlight_text(corpus.data[corpus.data['paperId'] == doc_id]['title'].item(),
                                 doc_highlights[doc_id]['title']),
            abstract=highlight_text(corpus.data[corpus.data['paperId'] == doc_id]['abstract'].item(),
                                    doc_highlights[doc_id]['abstract']),
        )
        for i, (doc_id, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))
        if i < max_result_count or max_result_count == -1
    ]
    yield documents
