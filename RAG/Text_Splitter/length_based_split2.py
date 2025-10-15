from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader("Gaming.pdf")
pages = loader.load()

print(f"Total Pages: {len(pages)}")


splitter=CharacterTextSplitter(
    separator="", chunk_size=500, chunk_overlap=10
)

docs=splitter.split_documents(pages)

# every chunk has pagecontent and metadata
print(f"Total Chunks: {len(docs)}")
print("chunk 0 : ",docs[0].page_content)
print("chunk 1 : ",docs[1].page_content)
print("chunk 2 :",docs[2].page_content)
print("chunk 3 :",docs[3].page_content)