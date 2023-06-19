from typing import Tuple

import numpy as np


def find_similar_docs(input_doc_index: int,
                      num_of_similar_docs: int,
                      emb_vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds similar documents to input in dataset using L2 distance

    Parameters
    ----------
    input_doc_index: int
        Index of input document vector in emb_vecs list to search for specific paper

    num_of_similar_docs:
        Number of similar documents to return

    emb_vecs : List
        A list of vectors corresponding to documents

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays, first array contains indices of similar documents
    and second array contains distances of similar documents
    """
    input_embedding = emb_vecs[input_doc_index]
    distances = np.linalg.norm(emb_vecs - input_embedding, axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices[1:num_of_similar_docs + 1], distances[sorted_indices[1:num_of_similar_docs + 1]]
