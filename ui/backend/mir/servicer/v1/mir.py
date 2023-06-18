from dataclasses import asdict
from typing import List

import grpc
import numpy as np

from mir.api.v1.mir_pb2_grpc import SearchServiceServicer
from mir.api.v1.mir_pb2 import AiBio,\
    Phase1SearchResponse, Phase1Paper,\
    Phase2SearchResponse, Phase2Paper,\
    Phase3SearchResponse, Phase3Paper,\
    RandomSimilarPapersResponse, SimilarPaper
from mir.clustering.similar import find_similar_docs
from mir.search.common import Phase1
from mir.search.search.phase1 import search as phase1_search, SearchResult as Phase1SearchResult
from mir.search.search.phase2 import search as phase2_search, SearchResult as Phase2SearchResult
from mir.search.search.phase3 import search as phase3_search, SearchResult as Phase3SearchResult
from mir.server import ai_bio, hw_system, arxiv, ss, similar_papers


class SearchService(SearchServiceServicer):

    @staticmethod
    def get_phase1(dataset) -> Phase1:
        return ai_bio if dataset == AiBio else hw_system

    def Phase1Search(self, request, context: grpc.ServicerContext):
        phase1 = self.get_phase1(request.dataset)
        search_handle = phase1_search(phase1.corpus,
                                      phase1.trie,
                                      phase1.bigram_index,
                                      request.query,
                                      request.max_result_count,
                                      highlight=True,
                                      method=request.ranking_method,
                                      weight=request.title_weight)
        corrected_query: str = next(search_handle)
        results: List[Phase1SearchResult] = next(search_handle)
        return Phase1SearchResponse(
            corrected_query=corrected_query,
            hits=[Phase1Paper(**asdict(result)) for result in results],
        )

    def Phase2Search(self, request, context: grpc.ServicerContext):
        phase2 = arxiv
        search_handle = phase2_search(phase2.corpus,
                                      phase2.trie,
                                      phase2.bigram_index,
                                      request.query,
                                      request.max_result_count,
                                      method=request.ranking_method,
                                      weight=request.title_weight,
                                      highlight=True,
                                      category=request.category,
                                      kmeans_dict=phase2.kmeans_dict)
        corrected_query: str = next(search_handle)
        results: List[Phase2SearchResult] = next(search_handle)
        return Phase2SearchResponse(
            corrected_query=corrected_query,
            hits=[Phase2Paper(**asdict(result)) for result in results],
        )

    def Phase3Search(self, request, context: grpc.ServicerContext):
        phase3 = ss
        search_handle = phase3_search(phase3.corpus,
                                      phase3.trie,
                                      phase3.bigram_index,
                                      request.query,
                                      request.max_result_count,
                                      method=request.ranking_method,
                                      weight=request.title_weight,
                                      highlight=True,
                                      personalization_weight=request.personalization_weight,
                                      preference_by_professor=request.preference_by_professor)
        corrected_query: str = next(search_handle)
        results: List[Phase3SearchResult] = next(search_handle)
        return Phase3SearchResponse(
            corrected_query=corrected_query,
            hits=[Phase3Paper(**asdict(result)) for result in results],
        )

    @staticmethod
    def __get_similar_papers(paper_id: str) -> SimilarPaper:
        return SimilarPaper(
            doc_id=paper_id,
            title=similar_papers.data.loc[int(paper_id), "titles"],
            category=similar_papers.data.loc[int(paper_id), "category"],
            abstract=similar_papers.data.loc[int(paper_id), "abstracts"],
        )

    def RandomSimilarPapers(self, request, context: grpc.ServicerContext):
        paper_id = np.random.choice(similar_papers.data['paper_id'].tolist())
        result = find_similar_docs(int(paper_id), request.number_of_similars, similar_papers.docs_embedding)
        return RandomSimilarPapersResponse(
            query_paper=self.__get_similar_papers(paper_id),
            similar_papers=[self.__get_similar_papers(paper_id) for paper_id in result]
        )
