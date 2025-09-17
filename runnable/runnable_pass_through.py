from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

#Defining model
model = ChatHuggingFace(llm = huggingfaceEndpointObject)

#Defining parser
parser = StrOutputParser()

#Defining prompt
prompt1 = PromptTemplate(template='write down the name top scorer in fotball for  {country}. Only name and nothing more.', 
                         input_variables=['country'])

prompt2 = PromptTemplate(template='Write one line for  {player}', 
                         input_variables=['player'])

#Defining chains
chain1 = RunnableSequence(prompt1 , model , parser)

chain2 = RunnableParallel({
    'name' : RunnablePassthrough(),
    'one_liner' : RunnableSequence(prompt2 , model , parser)
})

chain3 = RunnableSequence(chain1 , chain2)

#invoking chain
result = chain3.invoke({'country':'germany'})

print(result)