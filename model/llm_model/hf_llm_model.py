from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llmObject = OpenAI(model="gpt-3.5-turbo-instruct")
result = llmObject.invoke("who is the prime minister of India")

print(result)

