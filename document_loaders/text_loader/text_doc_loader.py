from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=huggingfaceEndpointObject)

prompt = PromptTemplate(template='What is capital of {name}',
                        input_variables=['name'])

loader = TextLoader(file_path='Last_CC_Bill.txt', encoding='utf-8')
document = loader.load()

#print(document)

parser= StrOutputParser()
chain = prompt | model | parser
result = chain.invoke({'name':document[0].page_content})

print(result)


