#https://www.youtube.com/watch?v=3TGqlQxpuU0
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=huggingfaceEndpointObject)

chat_history = [
    SystemMessage(content="This is a simple Chat Bot")
]

while True:
    user_input = input('You :')
    human_msg = HumanMessage(content=user_input)
    print(user_input)
    chat_history.append(human_msg)
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    #chat_history.append(result.content)
    print('AI :',result.content)
    ai_msg = AIMessage(content=result.content)
    chat_history.append(ai_msg)
#print(chat_history)


    



#print(result.content) #should be the response back to the user