import json
from enum import Enum
from typing import List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Search Engine", page_icon="🔍")

from clustering.clustering import cluster_kmeans
from clustering.embeddings import load_docs_embedding
from clustering.preprocess import preprocess_text
from search.search.phase1 import search as search_phase1, SearchResult as SearchResultPhase1
from search.search.phase2 import search as search_phase2, SearchResult as SearchResultPhase2
from search.search.phase3 import search as search_phase3, SearchResult as SearchResultPhase3
from search.bigram_index.phase1 import create_bigram_index as create_bigram_index_phase1
from search.bigram_index.phase2 import create_bigram_index as create_bigram_index_phase2
from search.bigram_index.phase3 import create_bigram_index as create_bigram_index_phase3
from search.trie.phase1 import construct_positional_indexes as construct_positional_indexes_phase1
from search.trie.phase2 import construct_positional_indexes as construct_positional_indexes_phase2
from search.trie.phase3 import construct_positional_indexes as construct_positional_indexes_phase3
from search.corpus.phase1 import Corpus as CorpusPhase1
from search.corpus.phase2 import Corpus as CorpusPhase2
from search.corpus.phase3 import Corpus as CorpusPhase3


class Dataset(Enum):
    AiBio = "AI & Bioinformatics", "ai-bio"
    HwSystem = "Hardware & System", "hardware-system"
    ArXiv = "arXiv", "arxiv"
    SemanticScholar = "Semantic Scholar", "ss"


class RankingMethod(Enum):
    LtnLnn = 'ltn-lnn'
    LtcLnc = 'ltc-lnc'
    OkApi25 = 'okapi25'


class Phase2Categories(Enum):
    All = 'all'
    CsCv = 'cs.CV'
    CsLg = 'cs.LG'
    StatMl = 'stat.ML'


class Phase3Professors(Enum):
    Kasaei = 'Kasaei'
    Rabiee = 'Rabiee'
    Rohban = 'Rohban'
    Sharifi = 'Sharifi'
    Soleymani = 'Soleymani'


st.markdown("# Search Engine")
st.sidebar.header("Search Settings")

dataset = Dataset[st.sidebar.selectbox("Select dataset", Dataset.__members__, format_func=lambda x: Dataset[x].value[0])]

st.write('Searching in:', dataset.value[0])


@st.cache_resource
def load_data(dataset: Dataset):
    progress_bar = st.sidebar.progress(0)
    if dataset in (Dataset.AiBio, Dataset.HwSystem):
        corpus = CorpusPhase1(dataset_name=dataset.value[1], stop_topk=20, progress_bar=progress_bar)
    elif dataset == Dataset.ArXiv:
        corpus = CorpusPhase2('../data/arxiv/arxiv_data_210930-054931.csv', stop_topk=20, progress_bar=progress_bar)
    elif dataset == Dataset.SemanticScholar:
        papers_by_id = {}
        papers_by_professor = {}
        for professor in Phase3Professors:
            with open(f'../results/crawled_paper_{professor.value}.json') as f:
                papers_by_professor[professor.value] = json.load(f)
                for paper in papers_by_professor[professor.value]:
                    papers_by_id[paper['id']] = paper
        all_papers = list(papers_by_id.values())
        corpus = CorpusPhase3(papers_by_professor, all_papers, stop_topk=20, progress_bar=progress_bar)
    else:
        raise ValueError(f'Unknown dataset {dataset}')
    _ = corpus.non_stop_documents
    progress_bar.empty()
    return corpus


@st.cache_data
def process_arxiv_data(_data: pd.DataFrame) -> pd.Series:
    return _data['titles'].str.cat(_data['abstracts'], sep=' ') \
        .apply(preprocess_text).str.join(' ')


info = st.sidebar.info('Processing documents')
corpus = load_data(dataset)

if corpus:
    if dataset in (Dataset.AiBio, Dataset.HwSystem):
        info.info('Constructing positional indices')
        trie = construct_positional_indexes_phase1(dataset.value[0], corpus)
        info.info('Constructing bigram indices')
        bigram_index = create_bigram_index_phase1(dataset.value[0], corpus)
        info.success(f'Loaded {len(corpus)} documents')

        max_result_count = st.sidebar.slider('Max results', 1, 100, 10)
        ranking_method = RankingMethod[
            st.sidebar.selectbox('Ranking method', RankingMethod.__members__, format_func=lambda x: RankingMethod[x].value)]
        title_weight = st.sidebar.slider('Title weight', 0.0, 1.0, 0.5)

        query = st.sidebar.text_input('Enter your query here')
        search_button = st.sidebar.button('Search')

        if search_button:
            with st.spinner('Searching...'):
                search_handle = search_phase1(corpus, trie, bigram_index, query, max_result_count,
                                              method=ranking_method.value,
                                              highlight=True,
                                              weight=title_weight)
                corrected_query: str = next(search_handle)
                st.markdown(f'## Query: {corrected_query}')
                results: List[SearchResultPhase1] = next(search_handle)
            for i, result in enumerate(results):
                st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
                st.markdown(f'Score: {result.score}')
                st.markdown(f'#### Abstract')
                st.markdown(result.abstract, unsafe_allow_html=True)
                st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
                st.markdown('---')
    elif dataset == Dataset.ArXiv:
        info.info('Constructing positional indices')
        trie = construct_positional_indexes_phase2(dataset.value[0], corpus)
        info.info('Constructing bigram indices')
        bigram_index = create_bigram_index_phase2(dataset.value[0], corpus)
        info.info('Loading document embeddings')
        docs_embedding = load_docs_embedding('../drive/MyDrive/arxiv-sbert-embeddings.npy')[corpus.random_indices]
        info.info('Preprocessing the documents')
        preprocessed_documents = process_arxiv_data(corpus.data)
        info.info('Clustering the documents')
        kmeans_dict = cluster_kmeans(docs_embedding, preprocessed_documents, 3)
        info.success(f'Loaded {len(corpus)} documents')

        max_result_count = st.sidebar.slider('Max results', 1, 100, 10)
        ranking_method = RankingMethod[
            st.sidebar.selectbox('Ranking method', RankingMethod.__members__, format_func=lambda x: RankingMethod[x].value)]
        title_weight = st.sidebar.slider('Title weight', 0.0, 1.0, 0.5)
        category = Phase2Categories[
            st.sidebar.selectbox('Category', Phase2Categories.__members__, format_func=lambda x: Phase2Categories[x].value)]

        query = st.sidebar.text_input('Enter your query here')
        search_button = st.sidebar.button('Search')

        if search_button:
            with st.spinner('Searching...'):
                search_handle = search_phase2(corpus, trie, bigram_index, query, max_result_count,
                                              method=ranking_method.value,
                                              highlight=True,
                                              category=category.value,
                                              weight=title_weight)
                corrected_query: str = next(search_handle)
                st.markdown(f'## Query: {corrected_query}')
                results: List[SearchResultPhase2] = next(search_handle)
            for i, result in enumerate(results):
                st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
                st.markdown(f'Score: {result.score}')
                st.markdown(f'Category: {result.category}')
                st.markdown(f'Cluster: {kmeans_dict["main_topics"][kmeans_dict["cluster_indices"][int(result.doc_id)]]}')
                st.markdown(f'#### Abstract')
                st.markdown(result.abstract, unsafe_allow_html=True)
                st.markdown('---')

    elif dataset == Dataset.SemanticScholar:
        info.info('Constructing positional indices')
        trie = construct_positional_indexes_phase3(dataset.value[0], corpus)
        info.info('Constructing bigram indices')
        bigram_index = create_bigram_index_phase3(dataset.value[0], corpus)
        info.success(f'Loaded {len(corpus)} documents')

        max_result_count = st.sidebar.slider('Max results', 1, 100, 10)
        ranking_method = RankingMethod[
            st.sidebar.selectbox('Ranking method', RankingMethod.__members__, format_func=lambda x: RankingMethod[x].value)]
        title_weight = st.sidebar.slider('Title weight', 0.0, 1.0, 0.5)
        personalization_weight = st.sidebar.slider('Personalization weight', 0.0, 1.0, 0.5)
        preference_by_professor = {
            professor.value: st.sidebar.slider(f'Preference for {professor.value}', 0.0, 1.0, 0.5)
            for professor in Phase3Professors
        }

        query = st.sidebar.text_input('Enter your query here')
        search_button = st.sidebar.button('Search')

        if search_button:
            with st.spinner('Searching...'):
                search_handle = search_phase3(corpus, trie, bigram_index, query, max_result_count,
                                              method=ranking_method.value,
                                              highlight=True,
                                              weight=title_weight,
                                              personalization_weight=personalization_weight,
                                              preference_by_professor=preference_by_professor)
                corrected_query: str = next(search_handle)
                st.markdown(f'## Query: {corrected_query}')
                results: List[SearchResultPhase3] = next(search_handle)
            for i, result in enumerate(results):
                st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
                st.markdown(f'Score: {result.score} - '
                            f'Search Score: {result.search_score} - '
                            f'Pagerank Score: {result.pagerank_score}')
                st.markdown(f'#### Abstract')
                st.markdown(result.abstract, unsafe_allow_html=True)
                st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
                st.markdown('---')
