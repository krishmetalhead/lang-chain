from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnableLambda
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
#Defining function
def word_count(val):
    return len(val.split())

#Defining model
model = ChatHuggingFace(llm = huggingfaceEndpointObject)

#Defining prompt
prompt1 = PromptTemplate(template='Write 1 line about {name}', input_variables=['name'])
prompt2 = PromptTemplate(template='What is total goal scored by  {name}', input_variables=['name'])

#Defining parser
parser = StrOutputParser()

#Defining chains
chain1 = RunnableSequence(prompt1,model,parser)
chain2 = RunnableSequence(prompt2,model,parser)
chain3 = RunnableLambda(word_count)

chain4 = RunnableParallel({
    'content': chain2,
    'word_count': chain3
})

chain5 = RunnableSequence(chain1,chain4)

result = chain5.invoke('Miroslav Kolse')

print(result)

