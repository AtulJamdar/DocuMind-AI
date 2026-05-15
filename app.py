from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.vectordb import create_vector_db
import os

load_dotenv()
if os.path.exists("chroma_db"):
    print("Loading existing vector database from 'chroma_db' directory...")
    vector_db = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))    
else:
    docs = load_pdf("data/ReactJS.pdf")
    chunks = split_documents(docs)
    vector_db = create_vector_db(chunks)

results = vector_db.max_marginal_relevance_search(
    "What is ReactJS?",
    k=3
)

for i, result in enumerate(results):
    print(f"\nRESULT {i+1}")
    print(result.page_content)