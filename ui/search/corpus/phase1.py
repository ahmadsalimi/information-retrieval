import os
from functools import lru_cache
from typing import Union, Literal, Dict, List, Set

import numpy as np
import pandas as pd
import streamlit as st

from search.preprocess import batch_clean_data, find_stop_words
from search.token import Token


class Corpus:

    def __init__(self, dataset_name: Union[Literal['ai-bio'], Literal['hardware-system']], stop_topk: int = 30,
                 progress_bar: st.delta_generator.DeltaGenerator = None,
                 sample_max_size: int = 3000):
        self.data = self.load_data(dataset_name)
        self.random_indices = np.random.choice(len(self.data), min(len(self.data), sample_max_size), replace=False)
        self.data = self.data.iloc[self.random_indices].reset_index(drop=True)
        self.stop_topk = stop_topk
        self.progress_bar = progress_bar

    @staticmethod
    def load_data(dataset_name: Union[Literal['ai-bio'], Literal['hardware-system']]) -> pd.DataFrame:
        return pd.read_csv(os.path.join('..', 'data', dataset_name, 'data.csv'))[['paperId', 'title', 'abstract']]\
            .fillna('')

    @property
    @lru_cache
    def cleaned_documents(self) -> Dict[str, Dict[str, List[Token]]]:
        length = len(self.data) * 2
        iteration_per_progress = length // 100
        iter = 0

        def progress(x):
            if self.progress_bar is not None:
                nonlocal iter
                iter += 1
                if iter % iteration_per_progress == 0:
                    self.progress_bar.progress(iter // iteration_per_progress)
            return x

        cleaned_titles = list(map(progress, batch_clean_data(self.data['title'].tolist())))
        cleaned_abstracts = list(map(progress, batch_clean_data(self.data['abstract'].tolist())))

        return {
            paper_id: {
                'title': cleaned_title,
                'abstract': cleaned_abstract,
            }
            for paper_id, cleaned_title, cleaned_abstract in zip(
                self.data['paperId'],
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
