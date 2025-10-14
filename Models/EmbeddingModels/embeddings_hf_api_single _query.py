from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")   

text="LangChain is a framework for developing applications powered by language models."
vector=embeddings.embed_query(text)
print(vector)


