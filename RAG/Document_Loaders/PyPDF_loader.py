
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


def get_pypdf_loader_class():

	try:
		from langchain.document_loaders import PyPDFLoader
		return PyPDFLoader

	except Exception:
		raise RuntimeError(
				"No PyPDFLoader available. Install langchain or langchain-community."
		)


def load_pdf(path: str):
	Loader = get_pypdf_loader_class()
	loader = Loader(path)
	docs = loader.load()
	return docs


def main():
	base = Path(__file__).parent
	pdf_path = base / "LangChain_FAQ.pdf"
	if not pdf_path.exists():
		print(f"PDF not found at {pdf_path}. Place a PDF named LangChain_FAQ.pdf next to this file.")
		return

	try:
		docs = load_pdf(str(pdf_path))
	except Exception as e:
		print("Failed to load PDF:", e)
		return

	print(f"Loaded {len(docs)} document chunks/pages.")
	# Print a short preview of the first document/page
	first = docs[0]
	# Some loader versions return Document objects with page_content and metadata
	content = getattr(first, "page_content", str(first))
	print("--- Preview (first chunk) ---")
	print(content[:1000])
	print("--- Metadata (if available) ---")
	print(getattr(first, "metadata", {}))


if __name__ == "__main__":
	main()





# from langchain_community.document_loaders import PyPDFLoader

# loader=PyPDFLoader("LangChain_FAQ.pdf")
# docs=loader.load()

# print(docs[0].page_content)

# print(docs[1].metadata)