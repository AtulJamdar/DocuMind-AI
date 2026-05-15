import os
import sys
import streamlit as st
import tempfile 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.loader import load_pdf   
from utils.splitter import split_documents
from utils.vectordb import create_vector_db
from utils.retriever import get_retriever
from utils.rag_chain import create_rag_chain


st.set_page_config(
    page_title="DocuMind AI",
     page_icon="🤖"
)

st.title("🤖 DocuMind AI")
st.subheader("AI-Powered PDF Q&A System")

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    st.success("PDF uploaded successfully.")

    with st.spinner("Processing PDF..."):

        docs = load_pdf(temp_pdf_path)

        chunks = split_documents(docs)

        vector_db = create_vector_db(chunks)

        retriever = get_retriever(vector_db)

        rag_chain = create_rag_chain(retriever)

        st.session_state.rag_chain = rag_chain

    st.success("PDF processed successfully!")

if st.session_state.rag_chain:

    user_question = st.text_input(
        "Enter your question about the PDF:"
    )

    # if user_question:

    #     with st.spinner("Generating answer..."):

    #          response = st.session_state.rag_chain.invoke({
    #              "input": user_question
    #          })

    #     st.subheader("Answer:")
    #     st.write(response["answer"])

    
    if user_question:
        with st.spinner("Generating answer..."):
            response = rag_chain.invoke(user_question)
        
        st.subheader("Answer:")
        st.write(response)