import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Type

import grpc
import streamlit as st

from api.v1.mir_pb2 import Phase1SearchRequest, Phase2SearchRequest, Phase3SearchRequest, AiBio, HwSystem, Phase1Paper, \
    Phase2Paper, Phase3Paper
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


class SearchClient(ABC):

    def __init__(self, dataset: Dataset,
                 ranking_method: RankingMethod,
                 title_weight: float,
                 max_result_count: int,
                 query_placeholder: st.delta_generator.DeltaGenerator):
        self.dataset = dataset
        self.ranking_method = ranking_method
        self.title_weight = title_weight
        self.max_result_count = max_result_count
        self.query_placeholder = query_placeholder
        self.extra_inputs = self._create_extra_inputs()

    def _create_extra_inputs(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def _call_search(self, stub: SearchServiceStub, query: str):
        pass

    @abstractmethod
    def _show_result(self, result: Any):
        pass

    def search(self, query: str):
        with grpc.insecure_channel('backend:50051') as channel:
            t1 = time.time()
            stub = SearchServiceStub(channel)
            response = self._call_search(stub, query)
            t2 = time.time()
            st.success(f'Search done in {t2 - t1:.2f} seconds')
            corrected_query = ' '.join([corrected if corrected == actual else f'<b><i>{corrected}</i></b>'
                                        for corrected, actual
                                        in zip(response.corrected_query.split(" "), query.split(" "))])
            query_placeholder.markdown(f'## Query: {corrected_query}', unsafe_allow_html=True)
            for i, result in enumerate(response.hits):
                self._show_result(result)


class Phase1SearchClient(SearchClient):

    def _call_search(self, stub: SearchServiceStub, query: str):
        return stub.Phase1Search(Phase1SearchRequest(
            dataset=self.dataset.value[1],
            query=query,
            max_result_count=self.max_result_count,
            ranking_method=self.ranking_method.value,
            title_weight=self.title_weight,
            **self.extra_inputs,
        ))

    def _show_result(self, result: Phase1Paper):
        st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
        st.markdown(f'Score: {result.score}')
        st.markdown(f'#### Abstract')
        st.markdown(result.abstract, unsafe_allow_html=True)
        st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
        st.markdown('---')


class Phase2SearchClient(SearchClient):

    def _create_extra_inputs(self) -> Dict[str, Any]:
        return dict(
            category=Phase2Categories[
                st.sidebar.selectbox('Category',
                                     Phase2Categories.__members__,
                                     format_func=lambda x: Phase2Categories[x].value)].value,
        )

    def _call_search(self, stub: SearchServiceStub, query: str):
        return stub.Phase2Search(Phase2SearchRequest(
            query=query,
            max_result_count=self.max_result_count,
            ranking_method=self.ranking_method.value,
            title_weight=self.title_weight,
            **self.extra_inputs,
        ))

    def _show_result(self, result: Phase2Paper):
        st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
        st.markdown(f'Score: {result.score}')
        st.markdown(f'Category: {result.category}')
        st.markdown(f'Cluster: {result.cluster}')
        st.markdown(f'#### Abstract')
        st.markdown(result.abstract, unsafe_allow_html=True)
        st.markdown('---')


class Phase3SearchClient(SearchClient):

    def _create_extra_inputs(self) -> Dict[str, Any]:
        personalization_weight = st.sidebar.slider('Personalization weight', 0.0, 1.0, 0.5)
        preference_by_professor = {
            professor.value: st.sidebar.slider(f'Preference for {professor.value}', 0.0, 1.0, 0.5)
            for professor in Phase3Professors
        }
        return dict(
            personalization_weight=personalization_weight,
            preference_by_professor=preference_by_professor,
        )

    def _call_search(self, stub: SearchServiceStub, query: str):
        return stub.Phase3Search(Phase3SearchRequest(
            query=query,
            max_result_count=self.max_result_count,
            ranking_method=self.ranking_method.value,
            title_weight=self.title_weight,
            **self.extra_inputs,
        ))

    def _show_result(self, result: Phase3Paper):
        st.markdown(f'### {i + 1}. {result.title}', unsafe_allow_html=True)
        st.markdown(f'Score: {result.score} - '
                    f'Search Score: {result.search_score} - '
                    f'Pagerank Score: {result.pagerank_score}')
        st.markdown(f'Topics: {", ".join(result.related_topics)}')
        st.markdown(f'#### Abstract')
        st.markdown(result.abstract, unsafe_allow_html=True)
        st.markdown(f'[open in Semantic Scholar](https://semanticscholar.org/paper/{result.doc_id})')
        st.markdown('---')


def get_search_client_class(dataset: Dataset) -> Type[SearchClient]:
    if dataset in (Dataset.AiBio, Dataset.HwSystem):
        return Phase1SearchClient
    if dataset == Dataset.ArXiv:
        return Phase2SearchClient
    if dataset == Dataset.SemanticScholar:
        return Phase3SearchClient
    raise ValueError(f'Unknown dataset {dataset}')


st.markdown("# Search Engine")
st.sidebar.header("Search Settings")

dataset = Dataset[st.sidebar.selectbox("Select dataset", Dataset.__members__, format_func=lambda x: Dataset[x].value[0])]

st.write('Searching in:', dataset.value[0])

max_result_count = st.sidebar.slider('Max results', 1, 100, 10)
ranking_method = RankingMethod[
    st.sidebar.selectbox('Ranking method', RankingMethod.__members__, format_func=lambda x: RankingMethod[x].value)]
title_weight = st.sidebar.slider('Title weight', 0.0, 1.0, 0.5)

col1, col2 = st.columns(2)
with col1:
    query = st.text_input('Enter your query here')

with col2:
    search_button = st.button('Search')

if query:
    query_placeholder = st.markdown(f'## Query: {query}')

search_client = get_search_client_class(dataset)(dataset, max_result_count, ranking_method, title_weight)

if search_button and query:
    search_client.search(query)
