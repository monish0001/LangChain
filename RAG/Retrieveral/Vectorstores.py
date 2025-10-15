from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")



documents = [
    Document(page_content="AI ethics concerns include bias, privacy violations, and lack of transparency."),
    Document(page_content="Explainability techniques help users understand model predictions and build trust."),
    Document(page_content="Retrieval-augmented generation (RAG) combines vector search with LLMs for grounded answers."),
    Document(page_content="Embeddings map text into vectors that allow semantic similarity and nearest-neighbor search."),
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="ai_collection"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

query = "What is RAG?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n Result {i}")
    print(doc.page_content)


