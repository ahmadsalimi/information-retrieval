from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

from mir.search.preprocess import batch_clean_data, find_stop_words
from mir.search.token import Token


class Corpus:

    def __init__(self, papers_by_professor: Dict[str, List[dict]],
                 all_papers: List[dict], stop_topk: int = 30):
        self.papers_by_professor = {
            professor: [paper['id'] for paper in papers]
            for professor, papers in papers_by_professor.items()
        }
        self.data = pd.DataFrame(all_papers)
        self.stop_topk = stop_topk
        self.cleaned_documents = self.get_cleaned_documents()
        self.stop_tokens = self.get_stop_tokens()
        self.non_stop_documents = self.get_non_stop_documents()

    def get_cleaned_documents(self) -> Dict[str, Dict[str, List[Token]]]:
        cleaned_titles = list(tqdm(batch_clean_data(self.data['title'].tolist()),
                                   total=len(self.data), desc='Cleaning Titles'))
        cleaned_abstracts = list(tqdm(batch_clean_data(self.data['abstract'].tolist()),
                                      total=len(self.data), desc='Cleaning Titles'))

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

    def get_stop_tokens(self) -> Set[str]:
        return find_stop_words(
            [token.processed
             for tokens in self.cleaned_documents.values()
             for token in tokens['title'] + tokens['abstract']],
            num_token=self.stop_topk,
        )

    def get_non_stop_documents(self) -> Dict[str, Dict[str, List[Token]]]:
        return {
            paper_id: {
                'title': [token for token in tokens['title'] if token.processed not in self.stop_tokens],
                'abstract': [token for token in tokens['abstract'] if token.processed not in self.stop_tokens],
            }
            for paper_id, tokens in self.cleaned_documents.items()
        }

    def __len__(self) -> int:
        return len(self.data)
