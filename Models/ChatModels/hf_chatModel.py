from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

chat = ChatHuggingFace(llm=llm)

response = chat.invoke([HumanMessage(content="Explain quantum computing in simple terms.")])
print(response.content)
