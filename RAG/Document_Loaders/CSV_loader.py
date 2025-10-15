from langchain_community.document_loaders import CSVLoader
file_path ="DMart_Transactions.csv"
loader = CSVLoader(file_path, encoding="utf-8")
docs = loader.load()
print(f"Loaded {len(docs)} documents.\n Preview:\n", docs[0])