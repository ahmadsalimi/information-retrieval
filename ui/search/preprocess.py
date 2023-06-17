from collections import Counter
from typing import List, Iterator, Set

import pandas as pd
import spacy
import streamlit as st

from search.token import Token


@st.cache_resource()
def get_nlp() -> spacy.Language:
    try:
        print('Loading spacy model')
        return spacy.load('en_core_web_md')
    finally:
        print('Loaded spacy model')


nlp = get_nlp()


def batch_clean_data(texts: List[str], batch_size: int = 128) -> Iterator[List[Token]]:
    """Preprocesses the text with tokenization, case folding, stemming and lemmatization, and punctuations

    Parameters
    ----------
    texts : List[str]
        A list of titles or abstracts of articles
    batch_size : int, optional
        The number of texts to be processed at a time, by default 128

    Returns
    -------
    List[List[Doc]]
        A list of lists of tokens
    """
    yield from ([Token.from_spacy_token(token) for token in doc if not token.is_punct]
                for doc in nlp.pipe(texts, batch_size=batch_size))


def clean_data(text: str) -> List[Token]:
    """Preprocesses the text with tokenization, case folding, stemming and lemmatization, and punctuations

    Parameters
    ----------
    text : str
        The title or abstract of an article

    Returns
    -------
    list
        A list of tokens
    """
    return next(batch_clean_data([text]))


def find_stop_words(all_text: List[str], num_token: int = 30) -> Set[str]:
    """Detects stop-words

     Parameters
    ----------
    all_text : list of all tokens
        (result of clean_data(text) for all the text)

    num_token : int
        number of stop words to be detected

    Returns
    -------
    Return Value is optional but must print the stop words and number of their occurence
    """
    counter = Counter(all_text)
    most_occur = counter.most_common(num_token)
    print(pd.DataFrame(most_occur, columns=['token', 'count']))
    return set([token for token, _ in most_occur])
