from typing import Dict, Literal, Set, Optional, Iterable, Tuple

import streamlit as st

from search.corpus.phase3 import Corpus
from search.token import Token


class TrieNode:

    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.idx_in_doc: Dict[str, Dict[Literal['title', 'abstract'], Set[int]]] = {}
        self.is_end = False

    def insert(self, doc_id: str, doc_section: Literal['title', 'abstract'], token: Token):
        node = self
        for char in token.processed:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            if doc_id not in node.idx_in_doc:
                node.idx_in_doc[doc_id] = {'title': set(), 'abstract': set()}
            node.idx_in_doc[doc_id][doc_section].add(token.idx)
        node.is_end = True

    def search(self, token: str) -> Optional[Dict[str, Dict[Literal['title', 'abstract'], Set[int]]]]:
        node = self
        for char in token:
            if char not in node.children:
                return None
            node = node.children[char]
        if node.is_end:
            return node.idx_in_doc
        return None

    def remove_document(self, doc_id: str):
        if doc_id in self.idx_in_doc:
            del self.idx_in_doc[doc_id]
        for child in self.children.values():
            child.remove_document(doc_id)

    def traverse_words(self, prefix: str = '') -> Iterable[Tuple[str, Dict[str, Dict[Literal['title', 'abstract'], Set[int]]]]]:
        if self.is_end:
            yield prefix, {
                doc_id: {
                    k: list(v) for k, v in doc.items()
                }
                for doc_id, doc in self.idx_in_doc.items()
            }

        for char, child in self.children.items():
            yield from child.traverse_words(prefix + char)

    @classmethod
    def from_words(cls, words: Iterable[Tuple[str, Dict[str, Dict[Literal['title', 'abstract'], Iterable[int]]]]]) -> 'TrieNode':
        root = cls()
        for word, idx_in_doc in words:
            for doc_id, doc in idx_in_doc.items():
                for doc_section, indices in doc.items():
                    for idx in indices:
                        token = Token(word, '', -1, idx)
                        root.insert(doc_id, doc_section, token)
        return root

    def to_dict(self):
        return {
            'children': {char: child.to_dict() for char, child in self.children.items()},
            'idx_in_doc': self.idx_in_doc,
            'is_end': self.is_end,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        node = cls()
        node.children = {char: cls.from_dict(child) for char, child in data['children'].items()}
        node.i_in_doc = data['i_in_doc']
        node.is_end = data['is_end']
        return node


@st.cache_resource
def construct_positional_indexes(dataset: str, _corpus: Corpus):
    """
    Get processed data and insert words in that into a trie and construct postional_index and posting lists after wards.

    Parameters
    ----------
    corpus: Corpus
        processed data

    Return
    ----------
    docs:
        list of docs with specificied id, title, abstract.
    """
    progress_bar = st.sidebar.progress(0)

    length = sum(sum(len(tokens) for tokens in doc.values()) for doc in _corpus.non_stop_documents.values())
    iteration_per_progress = length // 100
    i = 0

    def progress():
        if progress_bar is not None:
            nonlocal i
            i += 1
            if i % iteration_per_progress == 0:
                progress_bar.progress(i // iteration_per_progress)

    trie = TrieNode()
    for doc_id, doc in _corpus.non_stop_documents.items():
        for doc_section, tokens in doc.items():
            for token in tokens:
                progress()
                trie.insert(doc_id, doc_section, token)

    progress_bar.empty()
    return trie
