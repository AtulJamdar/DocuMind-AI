# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def create_vector_db(chunks):
    # Removed the 'models/' prefix to let the SDK construct the proper path
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Converting chunks to embeddings and saving to ChromaDB... Please wait.")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="vector_db"
    )
    
    print("Vector database successfully created and saved locally!")
    return vector_store