#https://www.youtube.com/watch?v=y5EmRr1O1h4&t=989s
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field 

load_dotenv()
model = ChatOpenAI(model='gpt-4o')

class Review(BaseModel):
    summary:str = Field(description='A breif summary in less than 10 words') 
    sentiment:Literal['Pos','Neg'] = Field(description='Overall sentiment')
    pros:Optional[list] = Field(description='List down the pros if any')
    pros:Optional[list] = Field(description='List down the cons if any')


structured_model = model.with_structured_output(Review)
print(structured_model)

result = structured_model.invoke('The phone hardware is great but software has multiple issue. Display is not' \
'upto the mark. RAM has glitches. Camera is good, which takes ood snaps in daylight but lacks in night .Overall in the proce range , the phone lacks expected behaviour.')

print('Review Summary : ',result['summary'])
print('Overall Sentiment : ',result['sentiment'])
print('Pros : ',result['pros'])
print('Cons : ',result['cons'])
