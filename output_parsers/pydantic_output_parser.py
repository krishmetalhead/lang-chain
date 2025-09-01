#https://www.youtube.com/watch?v=Op6PbJZ5b2Q
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


load_dotenv()

# Define your desired data structure.
class Detail(BaseModel):
    country: str = Field(description="Name of the country")
    capital: str = Field(description="Capital of the country")

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

parser = PydanticOutputParser(pydantic_object=Detail)

prompt_object=PromptTemplate(
    template="What is the capital of India \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
    )


model = ChatHuggingFace(llm=huggingfaceEndpointObject)

chain = prompt_object | model | parser

print(chain.invoke({}))
