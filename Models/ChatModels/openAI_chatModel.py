from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

model=ChatOpenAI(model="gpt-4")

result=model.invoke("Explain quantum computing in simple terms.")

print(result)