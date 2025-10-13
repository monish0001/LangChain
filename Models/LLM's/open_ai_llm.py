
from dotenv import load_dotenv
import os
from langchain_openai import OpenAI 

load_dotenv()


openai_api_key = os.getenv("OPENAI_API_KEY")


llm = OpenAI(api_key=openai_api_key, temperature=0.7)

response = llm.invoke("Explain quantum computing in simple terms.")

print(response)
