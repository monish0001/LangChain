from langchain_community.document_loaders import TextLoader

file_path ="ai_text.txt"
loader = TextLoader(str(file_path), encoding="utf-8")
docs = loader.load()
print(f"Loaded {len(docs)} documents.\n Preview:\n", docs[0])



