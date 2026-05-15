import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_db(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating embeddings...")


    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    print("Vector database created successfully!")

    return vector_store