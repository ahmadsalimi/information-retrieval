import numpy as np
import pandas as pd
import streamlit as st

from clustering.similar import find_similar_docs

st.set_page_config(page_title="Similar Papers", page_icon="📄")

from clustering.embeddings import load_docs_embedding


@st.cache_data
def load_data():
    df = pd.read_csv('../data/arxiv/arxiv_data_210930-054931.csv').fillna('')
    df['category'] = df['terms'].apply(lambda l: eval(l)[0])
    df = df.drop_duplicates(['titles']).drop_duplicates(['abstracts']).reset_index(drop=True)
    df['paper_id'] = df.index.astype(str)
    return df


st.markdown("# Similar Papers")
st.sidebar.header("Search Settings")

info = st.sidebar.info('Loading document embeddings')
docs_embedding = load_docs_embedding('../drive/MyDrive/arxiv-sbert-embeddings.npy')
info.info('Loading the documents')
data = load_data()
info.success(f'Loaded {len(data)} documents')

number_of_similars = st.sidebar.slider('Number of similar papers', 1, 100, 20)
select_random = st.sidebar.button('Select Random Paper')

if select_random:
    paper_id = np.random.choice(data['paper_id'].tolist())
    st.markdown(f'## {data.loc[int(paper_id), "titles"]}')
    st.markdown(f'### Category: {data.loc[int(paper_id), "category"]}')
    st.markdown('### Abstract')
    st.markdown(data.loc[int(paper_id), 'abstracts'])

    similar_papers = find_similar_docs(int(paper_id), number_of_similars, docs_embedding)

    st.markdown('### Similar Papers')
    for i, paper_id in enumerate(similar_papers):
        st.markdown(f'#### {i + 1}. {data.loc[int(paper_id), "titles"]}')
        st.markdown(f'Category: {data.loc[int(paper_id), "category"]}')
        st.markdown('##### Abstract')
        st.markdown(data.loc[int(paper_id), 'abstracts'])
        st.markdown('---')
