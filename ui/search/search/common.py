import math
from typing import Dict, Iterable, Literal, List

from search.corpus.phase1 import Corpus
from search.preprocess import nlp


def logarithmic_w_tf(tf: Dict[str, float]) -> Dict[str, float]:
    return {
        token: 1 + math.log(token_tf) if token_tf > 0 else 0
        for token, token_tf in tf.items()
    }


def cosine_normalization(w: Iterable[float]) -> float:
    return math.sqrt(sum(w_i ** 2 for w_i in w))


def get_df(token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]]) -> Dict[str, int]:
    return {
        token: len(search_results)
        for token, search_results in token_search_results.items()
    }


def query_tf_idf(query_tokens: List[str],
                 token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]],
                 score_type: str) -> Dict[str, float]:
    tf = {
        token: query_tokens.count(token)
        for token in token_search_results
    }
    if score_type[0] == 'n':
        w_tf = tf
    elif score_type[0] == 'l':
        w_tf = logarithmic_w_tf(tf)
    else:
        raise ValueError(f'tf method {score_type[0]} not supported')

    if score_type[1] == 'n':
        w_idf = {token: 1 for token in token_search_results}
    else:
        raise ValueError(f'idf method {score_type[1]} not supported')

    if score_type[2] == 'n':
        normalization = 1
    elif score_type[2] == 'c':
        normalization = cosine_normalization(w_tf.values())
    else:
        raise ValueError(f'normalization method {score_type[1]} not supported')

    w = {
        token: w_tf[token] * w_idf[token] / normalization
        for token in token_search_results
    }
    return w


def doc_tf_idf(corpus: Corpus,
               doc_section: str,
               token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]],
               query_w: Dict[str, float],
               score_type: str):
    if score_type[1] == 't':
        df = get_df(token_search_results)
        w_idf = {
            token: math.log(len(corpus) / token_df)
            for token, token_df in df.items()
        }
    else:
        raise ValueError(f'idf method {score_type[1]} not supported')

    all_documents = set()
    for search_results in token_search_results.values():
        all_documents.update(search_results.keys())

    doc_scores = {}
    for doc_id in all_documents:
        doc_tf = {
            token: len(search_results[doc_id][doc_section])
            for token, search_results in token_search_results.items()
            if doc_id in search_results
        }
        if score_type[0] == 'l':
            doc_w_tf = logarithmic_w_tf(doc_tf)
        else:
            raise ValueError(f'tf method {score_type[0]} not supported')

        if score_type[2] == 'n':
            doc_normalization = 1
        elif score_type[2] == 'c':
            doc_normalization = cosine_normalization(doc_w_tf.values())
        else:
            raise ValueError(f'normalization method {score_type[1]} not supported')

        doc_w = {
            token: doc_w_tf[token] * w_idf[token] / doc_normalization
            for token in doc_w_tf
            if doc_normalization > 0
        }
        doc_scores[doc_id] = sum(
            query_w[token] * doc_w[token]
            for token in query_w
            if token in doc_w
        )
    return doc_scores


def tf_idf(corpus: Corpus, doc_section: Literal['title', 'abstract'],
           token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]],
           query_tokens: List[str], score_type: str) -> Dict[str, float]:
    doc_score_type, query_score_type = score_type.split('-')
    query_w = query_tf_idf(query_tokens, token_search_results, query_score_type)
    return doc_tf_idf(corpus, doc_section, token_search_results, query_w, doc_score_type)


def okapi25(corpus: Corpus, doc_section: Literal['title', 'abstract'],
            token_search_results: Dict[str, Dict[str, Dict[Literal['title', 'abstract'], List[int]]]],
            query_tokens: List[str], k1: float = 1.2, b: float = 0.75) -> Dict[str, float]:
    all_documents = set()
    for search_results in token_search_results.values():
        all_documents.update(search_results.keys())

    df = get_df(token_search_results)
    idf = {
        token: math.log((len(all_documents) - token_df + 0.5) / (token_df + 0.5) + 1)
        for token, token_df in df.items()
    }
    f = {
        token: {
            doc_id: len(doc[doc_section])
            for doc_id, doc in search_results.items()
        }
        for token, search_results in token_search_results.items()
    }
    dl = {
        doc_id: len(corpus.non_stop_documents[doc_id][doc_section])
        for doc_id in all_documents
    }
    avgdl = sum(
        len(corpus.non_stop_documents[doc_id][doc_section])
        for doc_id in all_documents
    ) / len(all_documents)

    doc_scores = {}
    for doc_id in all_documents:
        doc_scores[doc_id] = sum(
            idf[token] * (
                    f[token].get(doc_id, 0.0) * (k1 + 1)
            ) / (
                    f[token].get(doc_id, 0.0) + k1 * (1 - b + b * dl[doc_id] / avgdl)
            )
            for token in query_tokens
        )
    return doc_scores


def highlight_text(text: str, highlight_starts: List[int]) -> str:
    result = ''
    last_index = 0
    highlight_starts = sorted(highlight_starts)
    for i, highlight_start in enumerate(highlight_starts):
        next_start = highlight_starts[i + 1] if i + 1 < len(highlight_starts) else len(text)
        token = nlp(text[highlight_start:next_start])[0]
        highlight_length = len(token)
        result += text[last_index:highlight_start] + '<mark style="background-color: #FFFF00">' + \
            text[highlight_start:highlight_start + highlight_length] + '</mark>'
        last_index = highlight_start + highlight_length
    result += text[last_index:]
    return result
