from typing import Iterable, Dict

from nltk.metrics import distance as nltkd

from mir.search.corpus.phase3 import Corpus
from mir.search.preprocess import nlp
from mir.search.common import pickle_cache

WORD_BOUNDARY_CHAR = '¶'


def get_word_bigrams(word: str) -> Iterable[str]:
    """
    Returns the bigrams of the given word

    Parameters
    ----------
    word: str
        The word to get the bigrams from

    Returns
    -------
    list
        A list of bigrams
    """
    word = WORD_BOUNDARY_CHAR + word + WORD_BOUNDARY_CHAR
    return [word[i:i + 2] for i in range(len(word) - 1)]


@pickle_cache(args_for_hash=['dataset'])
def create_bigram_index(dataset: str, _corpus: Corpus) -> Dict[str, Dict[str, int]]:
    """
    Creates a bigram index for the spell correction

    Parameters
    ----------
    _corpus: Corpus
        The corpus to generate the bigram index from

    Returns
    -------
    dict
        A dictionary of bigrams and their occurence
    """
    bigram_index: Dict[str, Dict[str, int]] = {}
    seen_words = set()
    for doc_id, doc in _corpus.cleaned_documents.items():
        for doc_section in doc.values():
            for token in doc_section:
                if token.actual in seen_words:
                    continue
                seen_words.add(token.actual)
                for bigram in get_word_bigrams(token.actual):
                    if bigram not in bigram_index:
                        bigram_index[bigram] = {}
                    if token.actual not in bigram_index[bigram]:
                        bigram_index[bigram][token.actual] = 0
                    bigram_index[bigram][token.actual] += 1
    return bigram_index


def correct_text(bigram_index: Dict[str, Dict[str, int]], text: str, similar_words_limit: int = 20) -> str:
    """
    Correct the give query text, if it is misspelled

    Parameters
    ---------
    bigram_index: Dict[str, Dict[str, int]]
        The bigram index to search in
    text: str
        The query text
    similar_words_limit: int
        The number of similar words

    Returns
    ---------
    str
        The corrected form of the given text
    """
    corrected_text = ''.join(text)
    for token in nlp(text):
        word = token.text
        if token.is_punct:
            continue
        word_occurences: Dict[str, int] = {}
        for bigram in get_word_bigrams(word):
            for posting, occurence in bigram_index.get(bigram, {}).items():
                if posting not in word_occurences:
                    word_occurences[posting] = 0
                word_occurences[posting] += occurence
        jaccard_scores = {
            posting: word_occurence / (len(word) + len(posting) + 2 - word_occurence)
            for posting, word_occurence in word_occurences.items()
        }
        similar_words = sorted(jaccard_scores, key=jaccard_scores.get, reverse=True)[:similar_words_limit]
        min_edit_distance = float('inf')
        corrected_word = word
        for similar_word in similar_words:
            if (edit_distance := nltkd.edit_distance(similar_word, word)) < min_edit_distance:
                min_edit_distance = edit_distance
                corrected_word = similar_word
        corrected_text = corrected_text.replace(word, corrected_word)
    return corrected_text
