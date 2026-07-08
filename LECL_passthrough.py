import os  
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ-API-KEY")
if not api_key:
    raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or GROQ-API-KEY in your environment or .env file.")  

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    api_key=api_key,
)   

summary_chain = (
    PromptTemplate.from_template(
        """
        Summarize the following text in a single paragraph:
        {topic}
        """ 
    )
    | llm
    | StrOutputParser() 
)

chain = RunnableParallel(
    originaltopic = RunnablePassthrough(),
    summary = summary_chain
)  

result = chain.invoke({"topic": "LangChain"})
print("Original Topic:", result["originaltopic"])
print("Summary:", result["summary"])    

print(result)