import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ-API-KEY")
if not api_key:
    raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or GROQ-API-KEY in your environment or .env file.")

prompt = PromptTemplate.from_template(
    """
    Explain {topic} to a 5th grade student in single sentence.
    """
)  

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    api_key=api_key,
)    

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "AI"})
print(result)

#chain = model.create_chain(prompt=prompt, output_parser=parser)