import os
import sys

print("Starting DocuMind application...", flush=True)

if os.path.exists("chroma_db"):
    print("Vector database found. Loading embedding model (this may take a moment)...", flush=True)
    
    # Only import heavy ML libraries when actually needed
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from utils.retriever import get_retriever
    from utils.rag_chain import create_rag_chain
    
    print("Initializing embeddings...", flush=True)
    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("Loading vector database from disk...", flush=True)
    vector_db = Chroma(
        persist_directory="chroma_db",
        embedding_function=EMBEDDING_MODEL
    )
    print(f"✓ Vector database loaded successfully!", flush=True)
else:
    print("\n" + "="*60, flush=True)
    print("❌ No vector database found!", flush=True)
    print("="*60, flush=True)
    print("\nTo use this application, you need to:", flush=True)
    print("1. Run the Streamlit frontend app:", flush=True)
    print("   streamlit run frontend/streamlit_app.py", flush=True)
    print("2. Upload a PDF file in the browser interface", flush=True)
    print("3. The app will create and save the vector database", flush=True)
    print("4. Then you can use this CLI app", flush=True)
    print("="*60 + "\n", flush=True)
    vector_db = None

if vector_db is not None:
    retriever = get_retriever(vector_db)
    qa_chain = create_rag_chain(retriever)
    query = input("Enter your question: ")
    response = qa_chain.invoke(query)
    print("\nAnswer:\n")
    print(response)
else:
    print("No vector database loaded. Please create one first by running with a valid PDF.")