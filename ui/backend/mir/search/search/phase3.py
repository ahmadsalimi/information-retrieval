from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Literal, List, Iterator

from mir.search.bigram_index.phase3 import correct_text
from mir.search.corpus.phase3 import Corpus
from mir.search.preprocess import clean_data
from mir.search.search.common import tf_idf, okapi25, highlight_text
from mir.search.trie.phase3 import TrieNode


@dataclass
class SearchResult:
    doc_id: str
    score: float
    search_score: float
    pagerank_score: float
    title: str
    abstract: str
    related_topics: List[str]


def pagerank(graph: Dict[str, List[str]], user_preferences: Dict[str, float]) -> Dict[str, float]:
    """
    Returns the personalized PageRank scores for the nodes in the graph, given the user's preferences.

    Parameters:
    graph (Dict[str, List[str]]): The graph represented as a dictionary of node IDs and their outgoing edges.
    user_preferences (Dict[str, float]): A dictionary of node IDs and the user's preferences for those nodes.

    Returns:
    Dict[str, float]: A dictionary of node IDs and their personalized PageRank scores.
    """

    # Constants for the PageRank algorithm
    damping_factor = 0.85
    convergence_threshold = 0.0001
    max_iterations = 100

    # Initialize the PageRank scores with equal probabilities
    num_nodes = len(graph)
    initial_score = 1.0 / num_nodes
    pagerank_scores = {node: initial_score for node in graph}

    # Convert user preferences to personalized teleportation probabilities
    teleportation_probs = {}
    total_preference = sum(user_preferences.values())
    if total_preference > 0:
        for node, preference in user_preferences.items():
            teleportation_probs[node] = preference / total_preference

    incoming_graph = {}
    for node, outgoing_nodes in graph.items():
        for outgoing_node in outgoing_nodes:
            if outgoing_node not in incoming_graph:
                incoming_graph[outgoing_node] = []
            incoming_graph[outgoing_node].append(node)

    # Iteratively calculate the PageRank scores
    for _ in range(max_iterations):
        new_scores = {}
        for node in graph:
            new_score = (1 - damping_factor) / num_nodes

            # Consider incoming edges to calculate the new score
            for incoming_node in incoming_graph.get(node, []):
                new_score += damping_factor * pagerank_scores[incoming_node] / len(graph[incoming_node])

            # Apply personalized teleportation if available
            if node in teleportation_probs:
                new_score += (1 - damping_factor) * teleportation_probs[node]

            new_scores[node] = new_score

        # Check for convergence
        convergence = sum(abs(new_scores[node] - pagerank_scores[node]) for node in graph)
        if convergence < convergence_threshold:
            break

        pagerank_scores = new_scores

    return pagerank_scores


def search(corpus: Corpus,
           trie: TrieNode,
           bigram_index: Dict[str, Dict[str, int]],
           query: str, max_result_count: int,
           method: str = 'ltn-lnn',
           weight: float = 0.5,
           highlight: bool = False,
           personalization_weight: float = 0.5,
           preference_by_professor: Dict[str, float] = None) -> Iterator:
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
        preference_by_professor: Dict[str, float]
            A dictionary of professor names and their preference scores

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

    graph = {}
    for _, paper in corpus.data.iterrows():
        graph[paper['id']] = paper['references']

    # Define the user preferences based on the professor's seed papers
    user_preferences = defaultdict(float)
    for professor, preference in preference_by_professor.items():
        for paper_id in corpus.papers_by_professor[professor]:
            user_preferences[paper_id] += preference

    # Calculate the personalized PageRank scores
    pagerank_scores = pagerank(graph, user_preferences)

    # normalize tf-idf scores
    doc_scores_max = max(doc_scores.values())
    normalized_doc_scores = {
        doc_id: doc_score / doc_scores_max
        for doc_id, doc_score in doc_scores.items()
    }

    # normalize pagerank scores
    pagerank_scores_max = max(pagerank_scores.values())
    normalized_pagerank_scores = {
        doc_id: pagerank_score / pagerank_scores_max
        for doc_id, pagerank_score in pagerank_scores.items()
    }

    # combine the two scores
    combined_scores = {
        doc_id: normalized_doc_scores[doc_id] * (1 - personalization_weight) + \
                normalized_pagerank_scores[doc_id] * personalization_weight
        for doc_id in normalized_doc_scores
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
            search_score=normalized_doc_scores[doc_id],
            pagerank_score=normalized_pagerank_scores[doc_id],
            title=highlight_text(corpus.data[corpus.data['id'] == doc_id]['title'].item(),
                                 doc_highlights[doc_id]['title']),
            abstract=highlight_text(corpus.data[corpus.data['id'] == doc_id]['abstract'].item(),
                                    doc_highlights[doc_id]['abstract']),
            related_topics=corpus.data[corpus.data['id'] == doc_id]['related_topics'].item(),
        )
        for i, (doc_id, score) in enumerate(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True))
        if i < max_result_count or max_result_count == -1
    ]
    yield documents
