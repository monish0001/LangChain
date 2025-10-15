from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

embeddings=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")

docs = [
    Document(page_content="Python is a versatile programming language used for web development and data science."),
    Document(page_content="Django is a high-level web framework that enables rapid development of secure websites."),
    Document(page_content="Flask is a lightweight microframework for building web applications in Python."),
    Document(page_content="FastAPI provides modern Python web APIs with automatic documentation and validation."),
]


vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)


retriever = vectorstore.as_retriever(
    search_type="mmr",                 
    search_kwargs={"k": 2, "lambda_mult": 0.2}  
)

query = "What is Python?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n Result {i+1}")
    print(doc.page_content)
    