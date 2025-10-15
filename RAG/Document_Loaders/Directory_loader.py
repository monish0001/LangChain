from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Create a loader for all PDF files in the "docs" folder
loader = DirectoryLoader(
    path="docs",
    glob="*.pdf",
    loader_cls=PyPDFLoader,  # Explicitly tell it how to load PDFs
    silent_errors=True
)

# Load all documents
documents = loader.load()
print(f"Loaded {len(documents)} documents.")

print(documents[11].page_content)  # Print the content of the first document
print(documents[0].metadata)      # Print metadata of the first document
