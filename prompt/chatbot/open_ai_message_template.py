from langchain_openai import ChatOpenAI 
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o')

template = ChatPromptTemplate([
        ("system", "Order Related Queries Helpdesk"),  
        MessagesPlaceholder(variable_name='chat_history'),   
        ("human", '{query}')
    ])

chat_history = []

try:
    with open("chat_history.txt", "r") as file:
        content = file.readline()
        chat_history.append(content)
except FileNotFoundError:
    print("Error: The file 'my_file.txt' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


prompt = template.invoke({
    'chat_history':chat_history,
    'query': input('You : ')
})

#chat_history.append(prompt)

print(prompt)
chat_response = AIMessage(content=model.invoke(prompt))
print(chat_response.content)











