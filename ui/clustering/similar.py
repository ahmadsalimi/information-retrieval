import numpy as np


def find_similar_docs(input_doc_index: int,
                      num_of_similar_docs: int,
                      emb_vecs: np.ndarray):
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
    List
        A list of similar document indexes to input document
    """
    input_embedding = emb_vecs[input_doc_index]
    distances = np.linalg.norm(emb_vecs - input_embedding, axis=1)
    return np.argsort(distances)[1:num_of_similar_docs + 1]
