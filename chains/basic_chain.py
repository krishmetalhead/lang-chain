#https://www.youtube.com/watch?v=5hjrPILA3-8
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

template = PromptTemplate(template='tell 2 lines about {text}' , 
                          input_variables=['text'])

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=huggingfaceEndpointObject)

parser = StrOutputParser()
chain = template | model | parser
result = chain.invoke({'text':input('Enter : ')})
print(result)