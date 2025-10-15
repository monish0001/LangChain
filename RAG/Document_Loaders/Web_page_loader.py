import os
# Set USER_AGENT to identify your requests
os.environ["USER_AGENT"] = "LangChainBot/1.0 (+https://example.com)"

from langchain_community.document_loaders import WebBaseLoader

# URL to load
url = "https://www.flipkart.com/motorola-g96-5g-pantone-ashleigh-blue-128-gb/p/itm93452c0761719?pid=MOBHB3SZ2ZUQQQ9U&lid=LSTMOBHB3SZ2ZUQQQ9U5BKC6D&marketplace=FLIPKART&store=tyy%2F4io&spotlightTagId=default_BestsellerId_tyy%2F4io&srno=b_1_1&otracker=browse&fm=organic&iid=d61e8041-f41f-4b3c-9668-8045626b80cd.MOBHB3SZ2ZUQQQ9U.SEARCH&ppt=browse&ppn=browse&ssid=5bppozsuo00000001760504087490"

# Initialize loader
loader = WebBaseLoader(url)

# Load documents
docs = loader.load()

# Output
print(f"Loaded {len(docs)} documents.\nPreview:\n", docs[0].page_content)
