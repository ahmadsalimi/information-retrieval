from functools import lru_cache
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import streamlit as st

from search.preprocess import batch_clean_data, find_stop_words
from search.token import Token


class Corpus:

    def __init__(self, dataset_path: str, stop_topk: int = 30,
                 progress_bar: st.delta_generator.DeltaGenerator = None,
                 sample_max_size: int = 10000):
        self.data = self.load_data(dataset_path, sample_max_size)
        self.data = self.data.iloc[self.random_indices]
        self.stop_topk = stop_topk
        self.progress_bar = progress_bar

    def load_data(self, dataset_path: str, sample_max_size: int) -> pd.DataFrame:
        df = pd.read_csv(dataset_path).fillna('')
        df['category'] = df['terms'].apply(lambda l: eval(l)[0])
        df = df.drop_duplicates(['titles']).drop_duplicates(['abstracts']).reset_index(drop=True)
        self.random_indices = np.random.choice(len(df), min(len(df), sample_max_size), replace=False)
        df = df.iloc[self.random_indices].reset_index(drop=True)
        df['paper_id'] = df.index.astype(str)
        return df

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

        cleaned_titles = list(map(progress, batch_clean_data(self.data['titles'].tolist())))
        cleaned_abstracts = list(map(progress, batch_clean_data(self.data['abstracts'].tolist())))

        return {
            paper_id: {
                'title': cleaned_titles,
                'abstract': cleaned_abstracts,
                'category': category,
            }
            for paper_id, cleaned_titles, cleaned_abstracts, category in zip(
                self.data['paper_id'].tolist(),
                cleaned_titles,
                cleaned_abstracts,
                self.data['category'].tolist(),
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
                'category': tokens['category'],
            }
            for paper_id, tokens in self.cleaned_documents.items()
        }

    @property
    @lru_cache
    def document_ids_by_category(self) -> Dict[str, Set[str]]:
        return {
            category: set(self.data[self.data['category'] == category]['paper_id'].tolist())
            for category in self.data['category'].unique()
        }

    def __len__(self) -> int:
        return len(self.data)
