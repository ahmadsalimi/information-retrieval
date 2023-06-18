import grpc
import streamlit as st

from api.v1.mir_pb2 import RandomSimilarPapersRequest
from api.v1.mir_pb2_grpc import SearchServiceStub

st.set_page_config(page_title="Similar Papers", page_icon="📄")

st.markdown("# Similar Papers")
st.sidebar.header("Search Settings")

number_of_similars = st.sidebar.slider('Number of similar papers', 1, 100, 20)
select_random = st.sidebar.button('Select Random Paper')

if select_random:
    with grpc.insecure_channel('backend:50051') as channel:
        stub = SearchServiceStub(channel)
        response = stub.RandomSimilarPapers(RandomSimilarPapersRequest(number_of_similars=number_of_similars))
    st.markdown(f'## {response.query_paper.title}')
    st.markdown(f'Category: {response.query_paper.category}')
    st.markdown('### Abstract')
    st.markdown(response.query_paper.abstract)

    st.markdown('### Similar Papers')
    for i, similar_paper in enumerate(response.similar_papers):
        st.markdown(f'#### {i + 1}. {response.query_paper.title}')
        st.markdown(f'Category: {response.query_paper.category}')
        st.markdown('##### Abstract')
        st.markdown(response.query_paper.abstract)
        st.markdown('---')
