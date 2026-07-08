# LECL - Langchain Expression Chain Language
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ-API-KEY")
if not api_key:
    raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or GROQ-API-KEY in your environment or .env file.")  

parser = StrOutputParser()

summary_prompt = PromptTemplate.from_template(
    """
    Summarize the following text in a single sentence:
    {topic}
    """
) 

example_prompt = PromptTemplate.from_template(
    """
    Provide an example of {topic} in a single sentence:
    """
)   

quiz_prompt = PromptTemplate.from_template(
    """
    Create a 3 quiz questions about {topic}:
    """
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    api_key=api_key,
)


summary_chain = summary_prompt | llm | parser
example_chain = example_prompt | llm | parser
quiz_chain = quiz_prompt | llm | parser

parallel_chain = RunnableParallel({
    "summary": summary_chain,
    "example": example_chain,
    "quiz": quiz_chain
})   

result = parallel_chain.invoke({"topic": "Gen AI"})
print("Summary:", result["summary"])
print("Example:", result["example"])
print("Quiz:", result["quiz"])  