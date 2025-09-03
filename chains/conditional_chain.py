#https://www.youtube.com/watch?v=Op6PbJZ5b2Q
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch , RunnableLambda

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model=ChatHuggingFace(llm=huggingfaceEndpointObject)

class sentiment(BaseModel):
   sentiment:Literal['Positive','Negative'] = Field(description='Sentiment Analysis')

parser=PydanticOutputParser(pydantic_object=sentiment)

primary_prompt = PromptTemplate(template="Classify the sentiment of the follwing feedback text into Positive or Negative \n {feedback} \n {format_instruction}",
                        input_variables='feedback',
                        partial_variables={'format_instruction': parser.get_format_instructions()})


primary_chain = primary_prompt | model | parser

conditional_prompt_1 = PromptTemplate(template="Write appropiate response to this positive feedback \n {feedback}",
                         input_variables=['feedback'])

conditional_prompt_2 = PromptTemplate(template="Write appropiate response to this negative feedback \n {feedback}",
                         input_variables=['feedback'])

string_parser=StrOutputParser()

conditional_chain = RunnableBranch(
   (lambda x: x.sentiment == 'Positive' , conditional_prompt_1 | model | string_parser ),
   (lambda x: x.sentiment == 'Negative' , conditional_prompt_2 | model | string_parser ),
   RunnableLambda(lambda x: 'Could not load sentiment')
)

final_chain = primary_chain | conditional_chain

result = final_chain.invoke({'feedback': 'The Nokia 73 is an terrible budget phone.'})
print(result)
   




