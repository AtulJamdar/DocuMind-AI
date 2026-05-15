from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.vectordb import create_vector_db
import os

load_dotenv()
docs = load_pdf("data/ReactJS.pdf")
chunks = split_documents(docs)
vector_db = create_vector_db(chunks)

print("Vector database created and persisted successfully.")

# print(f"Total Pages Loaded: {len(docs)}")
# print(f"Total Chunks: {len(chunks)}")
# print("\n First Chunk:\n")
# print(chunks[0].page_content)  # Print the first 500 characters of the first chunk

# print("\n Second Chunk: \n")
# print(chunks[1].page_content)  # Print the first 500 characters of the second chunk

# print(type(chunks[0]))
# print(chunks[0].metadata)

# llm = ChatGoogleGenerativeAI(
#     model="models/gemini-2.5-flash",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# response = llm.invoke("What is Retrieval-Augmented Generation?")

# print(response.content)