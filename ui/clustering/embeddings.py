import numpy as np
import streamlit as st


@st.cache_data
def load_docs_embedding(path: str):
    return np.load(path)
