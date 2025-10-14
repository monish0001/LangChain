from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")   

documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It provides tools and abstractions to simplify the integration of LLMs into applications.",
    "LangChain supports various use cases, including chatbots, document analysis, and code generation."
]
    
vector=embeddings.embed_documents(documents)
print(vector)


