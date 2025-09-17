from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

#Defining model
model = ChatHuggingFace(llm = huggingfaceEndpointObject)

#Defining prompt
prompt1 = PromptTemplate(template='Write 1 line about {name}', input_variables=['name'])
prompt2 = PromptTemplate(template='What is total goal scored by  {name}', input_variables=['name'])

#Defining parser
parser = StrOutputParser()

#Defining parrelal runnable
parrelal_chain = RunnableParallel({
    'detail': RunnableSequence(prompt1, model , parser),
    'total_goals': RunnableSequence(prompt2, model , parser),
})

#Calling parrelal runnable
result = parrelal_chain.invoke({'name':'Jurgen Klinsman'})
print(result)







