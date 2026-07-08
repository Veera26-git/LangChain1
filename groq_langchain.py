import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ-API-KEY")
if not api_key:
    raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or GROQ-API-KEY in your environment or .env file.")

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    api_key=api_key,
)

response = model.invoke("What is the capital of France?")
print(response.content)