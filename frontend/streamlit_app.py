import streamlit as st

st.title("DocuMind AI")

question = st.text_input("Ask a question")

if question:
    st.write(f"You asked: {question}")