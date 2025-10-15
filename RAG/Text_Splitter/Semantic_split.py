from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize SemanticChunker
text_splitter = SemanticChunker(embeddings=embeddings)

# Convert plain text into Document objects
documents = [
    Document(
        page_content="""
        The Amazon rainforest, often called the lungs of the Earth, is home to millions of species of plants and animals, many of which are still undiscovered. Protecting this ecosystem is critical for maintaining global biodiversity and regulating the climate. Meanwhile, quantum computing is emerging as a revolutionary technology, promising to solve complex problems far beyond the capabilities of classical computers. Companies and research labs are racing to develop stable qubits and error-correcting algorithms. On a completely different note, the culinary traditions of Italy have influenced global cuisine for centuries. From handmade pasta to rich cheeses, Italian food emphasizes fresh ingredients and time-honored recipes. Additionally, advances in renewable energy, like solar and wind power, are helping reduce dependence on fossil fuels and combat climate change. Sports also play a vital role in human culture, with football, basketball, and cricket uniting communities, promoting teamwork, and inspiring young athletes worldwide. Finally, modern art movements, from abstract expressionism to digital installations, challenge conventional perceptions and encourage viewers to explore new perspectives.
        """
    )
]

# Split the documents into semantic chunks
chunks = text_splitter.split_documents(documents)

# Print results
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")
