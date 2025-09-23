from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=huggingfaceEndpointObject)

prompt = PromptTemplate(template='Please generate a 1 liner header for content as {text}',
                        input_variables='text')


url = 'https://en.wikipedia.org/wiki/Solar_eclipse'
loader = WebBaseLoader(
    web_path=url
)
parser = StrOutputParser()
text = loader.load()
chain = prompt | model | parser

result = chain.invoke({'text': text[0].page_content})
print('Header generated as : ',result)