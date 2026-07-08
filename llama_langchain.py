from langchain_ollama import ChatOllama

model = ChatOllama(model = "qwen2.5:3b", temperature=0, max_tokens=512)

response = model.invoke("What is the capital of France?")
print(response.content)