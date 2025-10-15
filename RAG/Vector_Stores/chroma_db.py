from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.schema import Document

# Create documents
doc1 = Document(
    page_content="Virat Kohli is an iconic cricketer and former captain of India. In IPL, he represents Royal Challengers Bangalore and is known for his fearless batting, consistency, and leadership on the field.",
    metadata={"team": "Royal Challengers Bangalore"}
)
doc2 = Document(
    page_content="Rohit Sharma, captain of Mumbai Indians, has won multiple IPL titles. Famous for his powerful batting and calm demeanor, he is also a mentor to younger players in the squad.",
    metadata={"team": "Mumbai Indians"}
)
doc3 = Document(
    page_content="MS Dhoni, known as Captain Cool, has a legendary career in both international cricket and IPL with Chennai Super Kings. He is celebrated for his finishing skills, sharp wicketkeeping, and tactical acumen.",
    metadata={"team": "Chennai Super Kings"}
)
doc4 = Document(
    page_content="Jasprit Bumrah, Mumbai Indians' ace fast bowler, is renowned for his yorkers, death-over expertise, and ability to bowl under pressure in crucial moments of the game.",
    metadata={"team": "Mumbai Indians"}
)
doc5 = Document(
    page_content="Ravindra Jadeja is a versatile all-rounder for Chennai Super Kings. His accurate spin bowling, aggressive batting, and outstanding fielding make him an indispensable part of the team.",
    metadata={"team": "Chennai Super Kings"}
)
doc6 = Document(
    page_content="KL Rahul is an explosive opener known for his elegant stroke play and consistency. He has been a key player for Lucknow Super Giants and previously for Kings XI Punjab.",
    metadata={"team": "Lucknow Super Giants"}
)
doc7 = Document(
    page_content="Shreyas Iyer, captain of Delhi Capitals, is known for his elegant batting and calm leadership. He plays crucial innings and is a dependable middle-order batsman.",
    metadata={"team": "Delhi Capitals"}
)
doc8 = Document(
    page_content="Andre Russell, representing Kolkata Knight Riders, is a hard-hitting all-rounder with devastating power-hitting ability and useful fast-medium bowling.",
    metadata={"team": "Kolkata Knight Riders"}
)
doc9 = Document(
    page_content="Sunil Narine, the mystery spinner from Kolkata Knight Riders, is known for his economical bowling and explosive batting at the top of the order.",
    metadata={"team": "Kolkata Knight Riders"}
)
doc10 = Document(
    page_content="Suryakumar Yadav is a versatile middle-order batsman for Mumbai Indians, famous for innovative shots and quick scoring ability in T20 cricket.",
    metadata={"team": "Mumbai Indians"}
)

documents = [doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8, doc9, doc10]

# Initialize Chroma vector store
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory='my_chroma_db',
    collection_name='Cricket_Teams'
)

# Add documents to vector store
result = vector_store.add_documents(documents)
print("Documents added:", result)

# Retrieve all documents with embeddings and metadata
result = vector_store.get(include=['embeddings', 'documents', 'metadatas'])
print("All documents in vector store:", len(result['documents']))

# Search with similarity score
print("\n--- Similarity Search ---")
result = vector_store.similarity_search_with_score(
    query='Who among these are a bowler?',
    k=2
)
for doc, score in result:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print(f"Team: {doc.metadata['team']}")
    print("---")

# Meta-data filtering
print("\n--- Metadata Filtering (Mumbai Indians) ---")
result = vector_store.similarity_search_with_score(
    query="batting",  # Added a query for better results
    filter={"team": "Mumbai Indians"},
    k=3
)
for doc, score in result:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
    print(f"Team: {doc.metadata['team']}")
    print("---")

# Get document IDs for update/delete operations
all_docs = vector_store.get()
if all_docs['ids']:
    print(f"\nDocument IDs: {all_docs['ids']}")
    
    # Example of update (commented out as we don't have specific IDs)
    # updated_doc = Document(
    #     page_content="Updated content here",
    #     metadata={"team": "Mumbai Indians"}
    # )
    # vector_store.update_document(document_id=all_docs['ids'][0], document=updated_doc)
    
    # Example of delete (commented out)
    # vector_store.delete(ids=[all_docs['ids'][0]])