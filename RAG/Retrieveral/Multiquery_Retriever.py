from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv  
from langchain.retrievers.multi_query import MultiQueryRetriever
load_dotenv()

all_docs = [
    Document(page_content="Daily walking enhances cardiovascular fitness and helps ease depression symptoms.", metadata={"category": "wellness"}),
    Document(page_content="Eating plenty of leafy vegetables and fruits supports body detoxification and promotes longevity.", metadata={"category": "nutrition"}),
    Document(page_content="Getting deep sleep is essential for repairing cells and maintaining emotional balance.", metadata={"category": "sleep"}),
    Document(page_content="Practicing mindfulness and slow breathing reduces stress hormones and sharpens mental focus.", metadata={"category": "mindfulness"}),
    Document(page_content="Staying hydrated by drinking enough water all day supports metabolism and sustains energy levels.", metadata={"category": "hydration"}),
    Document(page_content="Modern homes using solar panels help regulate electricity consumption and energy needs.", metadata={"category": "sustainability"}),
    Document(page_content="Python combines simplicity with strong functionality, making it widely used for system design.", metadata={"category": "technology"}),
    Document(page_content="Through photosynthesis, plants transform sunlight into energy for growth.", metadata={"category": "biology"}),
    Document(page_content="The FIFA World Cup 2022 took place in Qatar, attracting worldwide enthusiasm and attention.", metadata={"category": "sports"}),
    Document(page_content="Black holes distort spacetime and contain an enormous amount of gravitational power.", metadata={"category": "astronomy"}),
]

# Use local embeddings instead of endpoint
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# For the LLM, you need to use a Hugging Face Inference Endpoint
# Make sure you have HUGGINGFACEHUB_API_TOKEN in your .env file
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)
chat = ChatHuggingFace(llm=llm)

vectorstore = FAISS.from_documents(
    documents=all_docs, 
    embedding=embeddings
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=chat,
)

# Query
query = "How to stay energized and maintain physical and mental equilibrium?"

multiquery_results = multiquery_retriever.invoke(query)

for i, doc in enumerate(multiquery_results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Category: {doc.metadata['category']}")