import time
from enum import Enum

import grpc
import streamlit as st

from api.v1.mir_pb2 import Phase1SearchRequest, Phase2SearchRequest, Phase3SearchRequest, AiBio, HwSystem
from api.v1.mir_pb2_grpc import SearchServiceStub

st.set_page_config(page_title="Search Engine", page_icon="🔍")


class Dataset(Enum):
    AiBio = "AI & Bioinformatics", AiBio
    HwSystem = "Hardware & System", HwSystem
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

max_result_count = st.sidebar.slider('Max results', 1, 100, 10)
ranking_method = RankingMethod[
    st.sidebar.selectbox('Ranking method', RankingMethod.__members__, format_func=lambda x: RankingMethod[x].value)]
title_weight = st.sidebar.slider('Title weight', 0.0, 1.0, 0.5)

if dataset in (Dataset.AiBio, Dataset.HwSystem):
    query = st.sidebar.text_input('Enter your query here')
    search_button = st.sidebar.button('Search')

    if search_button:
        t1 = time.time()
        with st.spinner('Searching...'):
            with grpc.insecure_channel('backend:50051') as channel:
                stub = SearchServiceStub(channel)
                response = stub.Phase1Search(Phase1SearchRequest(
                    dataset=dataset.value[1],
                    query=query,
                    max_result_count=max_result_count,
                    ranking_method=ranking_method.value,
                    title_weight=title_weight,
                ))
        t2 = time.time()
        st.success(f'Search done in {t2 - t1:.2f} seconds')
        st.markdown(f'## Query: {response.corrected_query}')
        for i, result in enumerate(response.hits):
            st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
            st.markdown(f'Score: {result.score}')
            st.markdown(f'#### Abstract')
            st.markdown(result.abstract, unsafe_allow_html=True)
            st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
            st.markdown('---')

elif dataset == Dataset.ArXiv:
    category = Phase2Categories[
        st.sidebar.selectbox('Category', Phase2Categories.__members__, format_func=lambda x: Phase2Categories[x].value)]

    query = st.sidebar.text_input('Enter your query here')
    search_button = st.sidebar.button('Search')

    if search_button:
        t1 = time.time()
        with st.spinner('Searching...'):
            with grpc.insecure_channel('backend:50051') as channel:
                stub = SearchServiceStub(channel)
                response = stub.Phase2Search(Phase2SearchRequest(
                    query=query,
                    max_result_count=max_result_count,
                    ranking_method=ranking_method.value,
                    title_weight=title_weight,
                    category=category.value,
                ))
        t2 = time.time()
        st.success(f'Search done in {t2 - t1:.2f} seconds')
        st.markdown(f'## Query: {response.corrected_query}')
        for i, result in enumerate(response.hits):
            st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
            st.markdown(f'Score: {result.score}')
            st.markdown(f'Category: {result.category}')
            st.markdown(f'Cluster: {result.cluster}')
            st.markdown(f'#### Abstract')
            st.markdown(result.abstract, unsafe_allow_html=True)
            st.markdown('---')

elif dataset == Dataset.SemanticScholar:
    personalization_weight = st.sidebar.slider('Personalization weight', 0.0, 1.0, 0.5)
    preference_by_professor = {
        professor.value: st.sidebar.slider(f'Preference for {professor.value}', 0.0, 1.0, 0.5)
        for professor in Phase3Professors
    }

    query = st.sidebar.text_input('Enter your query here')
    search_button = st.sidebar.button('Search')

    if search_button:
        t1 = time.time()
        with st.spinner('Searching...'):
            with grpc.insecure_channel('backend:50051') as channel:
                stub = SearchServiceStub(channel)
                response = stub.Phase3Search(Phase3SearchRequest(
                    query=query,
                    max_result_count=max_result_count,
                    ranking_method=ranking_method.value,
                    title_weight=title_weight,
                    personalization_weight=personalization_weight,
                    preference_by_professor=preference_by_professor,
                ))
        t2 = time.time()
        st.success(f'Search done in {t2 - t1:.2f} seconds')
        st.markdown(f'## Query: {response.corrected_query}')
        for i, result in enumerate(response.hits):
            st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
            st.markdown(f'Score: {result.score} - '
                        f'Search Score: {result.search_score} - '
                        f'Pagerank Score: {result.pagerank_score}')
            st.markdown(f'Topics: {", ".join(result.related_topics)}')
            st.markdown(f'#### Abstract')
            st.markdown(result.abstract, unsafe_allow_html=True)
            st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
            st.markdown('---')
