from functools import lru_cache
from typing import Dict, List, Set

import pandas as pd
import streamlit as st

from search.preprocess import batch_clean_data, find_stop_words
from search.token import Token


class Corpus:

    def __init__(self, papers_by_professor: Dict[str, List[dict]],
                 all_papers: List[dict], stop_topk: int = 30,
                 progress_bar: st.delta_generator.DeltaGenerator = None):
        self.papers_by_professor = {
            professor: [paper['id'] for paper in papers]
            for professor, papers in papers_by_professor.items()
        }
        self.data = pd.DataFrame(all_papers)
        self.stop_topk = stop_topk
        self.progress_bar = progress_bar

    @property
    @lru_cache
    def cleaned_documents(self) -> Dict[str, Dict[str, List[Token]]]:
        length = len(self.data) * 2
        iteration_per_progress = length // 100
        i = 0

        def progress(x):
            if self.progress_bar is not None:
                nonlocal i
                i += 1
                if i % iteration_per_progress == 0:
                    self.progress_bar.progress(i // iteration_per_progress)
            return x

        cleaned_titles = list(map(progress, batch_clean_data(self.data['title'].tolist())))
        cleaned_abstracts = list(map(progress, batch_clean_data(self.data['abstract'].tolist())))

        return {
            paper_id: {
                'title': cleaned_title,
                'abstract': cleaned_abstract,
            }
            for paper_id, cleaned_title, cleaned_abstract in zip(
                self.data['id'].tolist(),
                cleaned_titles,
                cleaned_abstracts,
            )
        }

    @property
    @lru_cache
    def stop_tokens(self) -> Set[str]:
        return find_stop_words(
            [token.processed
             for tokens in self.cleaned_documents.values()
             for token in tokens['title'] + tokens['abstract']],
            num_token=self.stop_topk,
        )

    @property
    @lru_cache
    def non_stop_documents(self) -> Dict[str, Dict[str, List[Token]]]:
        return {
            paper_id: {
                'title': [token for token in tokens['title'] if token.processed not in self.stop_tokens],
                'abstract': [token for token in tokens['abstract'] if token.processed not in self.stop_tokens],
            }
            for paper_id, tokens in self.cleaned_documents.items()
        }

    def __len__(self) -> int:
        return len(self.data)
