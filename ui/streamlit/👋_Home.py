import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="👋",
)

st.write("# Welcome to Modern Information Retrieval! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ## ℹ️ About
    
    This is a demo for the course Modern Information Retrieval at Sharif University of Technology,
    created by [Ahmad Salimi](https://linkedin.com/in/ahmadsalimi).
    
    ## 💻 Source Code
    
    The source code containing the jupyter notebooks for the demos, the backend, and the frontend
    of this demo is available on [GitHub](https://github.com/ahmadsalimi/information-retrieval).
    
    Backend is implemented with python using `gRPC` and the frontend is implemented with python using `streamlit`.

    ## 📚 Datasets
    
    Four datasets collected from [Semantic Scholar](https://www.semanticscholar.org/) and [arXiv](https://arxiv.org/)
    are used in this demo:
    
    - AI & Bioinformatics from Semantic Scholar
    - Hardware & System from Semantic Scholar
    - Random arXiv papers from 2019
    - Crawled Semantic Scholar papers with the five papers of each of the following professors as seeds:
        - [Dr. Shohreh Kasaei](https://scholar.google.com/citations?user=mvx4PvgAAAAJ&hl=en)
        - [Dr. Hamid R. Rabiee](https://scholar.google.com/citations?user=rKDtrNgAAAAJ&hl=en)
        - [Dr. Mohammad Hossein Rohban](https://scholar.google.com/citations?user=pRyJ6FkAAAAJ&hl=en)
        - [Dr. Ali Sharifi-Zarchi](https://scholar.google.com/citations?user=GbJMZLIAAAAJ&hl=en)
        - [Dr. Mahdieh Soleymani](https://scholar.google.com/citations?user=S1U0KlgAAAAJ&hl=en)

    ## 🤖 Models

    The model used for clustering and finding similar papers is Sentence Transformer from `huggingface`
    (`sentence-transformers/all-mpnet-base-v2`).
    
    Search methods (`tf-idf` and `okapi25`) are implemented from scratch using `numpy`, and `pandas`.
    
    `spacy` and `nltk` are used for preprocessing.
    
    ## 📝 References
    
    - [gRPC: A high performance, open-source universal RPC framework](https://grpc.io/)
    - [Streamlit: The fastest way to build custom ML tools](https://streamlit.io/)
    - [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
    - [NumPy: the fundamental package for scientific computing with Python](https://numpy.org/)
    - [pandas: Powerful data structures for data analysis, time series, and statistics](https://pandas.pydata.org/)
    - [spaCy: Industrial-strength Natural Language Processing in Python](https://spacy.io/)
    - [Natural Language Toolkit](https://www.nltk.org/)
"""
)