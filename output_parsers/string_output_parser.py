#https://www.youtube.com/watch?v=Op6PbJZ5b2Q
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

prompt_object_1=PromptTemplate.from_template("Who is the father of Mughal King {'mughal_king'}")
prompt_object_2=PromptTemplate.from_template("Who is the wife  {'text'}")

model = ChatHuggingFace(llm=huggingfaceEndpointObject)
parser = StrOutputParser()

chain = prompt_object_1 | model | parser | prompt_object_2 | model | parser

result = chain.invoke('Akbar')
print(result)




