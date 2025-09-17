from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

#Defining model
model = ChatHuggingFace(llm = huggingfaceEndpointObject)

#Defining prompt
prompt = PromptTemplate(template='Write 1 line about {name}', input_variables=['name'])

#Defining parser
parser = StrOutputParser()

#Defining runnable
runnable = RunnableSequence(prompt, model , parser)

#Rinning runnable sequence
result = runnable.invoke({'name':'Klinsman'})

print(result)

