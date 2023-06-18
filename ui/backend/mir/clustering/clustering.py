import numpy as np
import pandas as pd

from tqdm import tqdm

from mir.clustering.preprocess import find_main_topic
from mir.util import pickle_cache


@pickle_cache(args_for_hash=['emb_vecs', 'n_clusters'])
def cluster_kmeans(emb_vecs: np.ndarray, _processed_documents: pd.Series, n_clusters: int):
    """Clusters input vectors using K-means method from scratch

    Parameters
    ----------
    emb_vecs : np.ndarray
        A list of vectors
    n_clusters : int
        Number of clusters

    Returns
    -------
    """

    # Initialize cluster centers randomly
    cluster_centers = emb_vecs[np.random.choice(np.arange(len(emb_vecs)), n_clusters, replace=False)]

    # Iteratively update cluster centers until convergence
    with tqdm(desc=f'K-means clustering with k={n_clusters}') as pbar:
        while True:
            # Assign each vector to the nearest cluster center
            distances = np.linalg.norm(emb_vecs[:, np.newaxis] - cluster_centers, axis=-1)
            cluster_indices = distances.argmin(axis=-1)

            # Update cluster centers
            new_centers = np.array([emb_vecs[cluster_indices == i].mean(axis=0) for i in range(n_clusters)])

            # Check convergence
            if np.allclose(cluster_centers, new_centers):
                break

            pbar.set_postfix({'loss': np.linalg.norm(cluster_centers - new_centers)})
            pbar.update()
            cluster_centers = new_centers

    cluster_documents = [_processed_documents[cluster_indices == i] for i in range(n_clusters)]
    main_topics = [find_main_topic(docs) for docs in cluster_documents]

    return dict(
        cluster_centers=cluster_centers,
        cluster_indices=cluster_indices,
        main_topics=main_topics
    )
