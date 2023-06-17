import string

import nltk
import numpy as np
from nltk import word_tokenize
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


@st.cache_data
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    return True


load_nltk_data()


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                    punctuation_removal=True):

    normalized_tokens = word_tokenize(text)

    if stopword_removal:
        # Remove stopwords in English and also the given domain stopwords
        stopwords = [x.lower() for x in nltk.corpus.stopwords.words('english')]
        domain_stopwords = [x.lower() for x in stopwords_domain]
        normalized_tokens = [word for word in normalized_tokens if word.lower() not in domain_stopwords + stopwords]

    if punctuation_removal:
        # Remove punctuations
        normalized_tokens = [word for word in normalized_tokens if word not in string.punctuation]

    if lower_case:
        # Convert everything to lowercase and filter based on a min length
        normalized_tokens = [word.lower() for word in normalized_tokens if len(word) > minimum_length]
    else:
        normalized_tokens = [word for word in normalized_tokens if len(word) > minimum_length]

    return normalized_tokens


def find_main_topic(cluster_documents):
    """Find the main topic of a cluster by analyzing its documents using LDA

    Parameters
    ----------
    cluster_documents : list
        List of documents belonging to a cluster

    Returns
    -------
    str
        The main topic of the cluster
    """
    # Create a document-term matrix
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    dtm = vectorizer.fit_transform(cluster_documents)

    # Apply LDA
    n_topics = 1  # Assuming we want to find the main topic
    lda = LatentDirichletAllocation(n_components=n_topics)
    lda.fit(dtm)

    # Get the main topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = np.array(feature_names)[np.argsort(lda.components_[0])][::-1]
    main_topic = ', '.join(topic_words[:5])  # Assuming we want the top 10 words as the main topic

    return main_topic
