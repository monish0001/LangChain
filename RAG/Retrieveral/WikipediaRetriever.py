from langchain_community.retrievers import WikipediaRetriever


# Define your query
query = "What are the major ethical concerns surrounding the deployment of AI systems, and how are they being addressed globally?"

# Initialize the retriever (optional: set language and top_k)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Get relevant Wikipedia documents
docs = retriever.invoke(query)




# Get relevant Wikipedia documents
docs = retriever.invoke(query)

# Print retrieved content
for i, doc in enumerate(docs):
    print(f"\n Result {i} : ")
    print(f"Content: {doc.page_content}") 

