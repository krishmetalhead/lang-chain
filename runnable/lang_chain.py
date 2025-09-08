#https://www.youtube.com/watch?v=u3b-W1NgYa4
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=huggingfaceEndpointObject)

prompt = PromptTemplate(template='What is the capital of {country}',
                        input_variables=['country'])

chain = LLMChain(llm=model,prompt=prompt) #Currently deprecated
result = chain.run('India') 
print(result)
