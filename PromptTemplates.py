############### Single - Turn Prompt Template ###############
from langchain_core.prompts import PromptTemplate

promt = PromptTemplate.from_template("What is the capital of {country}?")

formated_prompt = promt.format(country="France")

print(formated_prompt)


################# Dynamic / multi - Turn Prompt Template ###############

promt = PromptTemplate.from_template(
    """
    Create a {days}-day itinerary for a trip to {country}. 
    Budget: {budget}
    Language: {language}
    """
    )

final_prompt = promt.format(days=5, country="France", budget="$2000", language="English")
print(final_prompt) 


################# Chat prompt Template ###############
from langchain_core.prompts import ChatPromptTemplate

promt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         "You are a helpful assistant that translates {input_language} to {output_language}."),
        
        ("human",
         "{text}"),
    ]
)

messages = promt.format_messages(
    input_language="English", 
    output_language="Telugu ", 
    text="Hello, how are you?")

print(messages)

################ Multi - turn chat or conversation prompt template ###############
from langchain_core.messages import HumanMessage 

messages = [
    HumanMessage(content="Hello, Who is Virat Kohli?"),
    HumanMessage(content="Where was he born?"),
]       


################ Chat prompt + History Template ###############
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

promt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant"),
        ("human",
         "{question}"),
        MessagesPlaceholder(variable_name="history"),
    ]
)

history_messages = [
    HumanMessage(content="Hello, who is Virat Kohli?")
]

messages = promt.format_messages(
    question="Where was he born?", 
    history=history_messages)