import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ-API-KEY")
if not api_key:
    raise RuntimeError("Missing Groq API key. Set GROQ_API_KEY or GROQ-API-KEY in your environment or .env file.")      

prompt = PromptTemplate.from_template(
    """
    Analyze the restaurant review.
    review: {review}


    Return the output in the following JSON format:
    {{
        "sentiment": "positive/negative/neutral",
        "aspects": [
            {{
                "aspect": "food/service/ambience/price",
                "sentiment": "positive/negative/neutral"
            }},
        ],
    "summary": "A brief summary of the review",
    "rating": "A rating out of 5"
    }}
    """
)

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    api_key=api_key,
)
parser = JsonOutputParser() 

cahin = prompt | model | parser

result = cahin.invoke({"review": "The food was amazing, but the service was slow."})
print(result)
print(result["sentiment"])
print(result["aspects"])
print(result["summary"])
print(result["rating"])