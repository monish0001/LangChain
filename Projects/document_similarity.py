from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()




# Sample documents
documents = [
    "The cat sat on the mat.",
    "The dog barked at the mailman.",
    "The cat chased the mouse.",
    "The quick brown fox jumps over the lazy dog."
]

# Query document
query = "A cat is sitting on a mat."


embeddings=HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2")  
document_embeddings = embeddings.embed_documents(documents)
query_embeddings = embeddings.embed_query(query)


cosine_similarities = cosine_similarity([query_embeddings], document_embeddings).flatten()
print(cosine_similarities)
most_similar_index = cosine_similarities.argmax()
print("Cosine similarities:", cosine_similarities)
print("Most similar document index:", most_similar_index)
print("query:", query)
print("Most similar document:", documents[most_similar_index])






