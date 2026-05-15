from langchain_chroma import Chroma
# from langchain_chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.vectordb import create_vector_db
from utils.retriever import get_retriever
from utils.rag_chain import create_rag_chain
import os

EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if os.path.exists("chroma_db"):

    print("Loading existing vector database...")

    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=EMBEDDING_MODEL
    )

else:

    docs = load_pdf("data/ReactJS.pdf")

    chunks = split_documents(docs)

    vector_db = create_vector_db(chunks)

retriever = get_retriever(vector_db)
qa_chain = create_rag_chain(retriever)
query = input("Enter your question: ")
response = qa_chain.invoke(query)
print("\nAnswer:\n")
print(response)