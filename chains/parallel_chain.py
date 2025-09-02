#https://www.youtube.com/watch?v=5hjrPILA3-8
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel

load_dotenv()


try:
    with open("input.txt", "r") as file:
        content = file.read()        
except FileNotFoundError:
    print("Error: The file 'my_file.txt' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

#print(content)
parser = StrOutputParser()
huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=huggingfaceEndpointObject)

prompt1 = PromptTemplate(template='Suumarize {content}',
                         input_variables=['content'])

prompt2 = PromptTemplate(template='Generate 2 quiz question from {content}',
                         input_variables=['content'])

parellael_chain=RunnableParallel({
    'chain1' : prompt1 | model | parser,
    'chain2' : prompt2 | model | parser
})

prompt3 = PromptTemplate(template='Merge {chain1} and {chain2}',
                         input_variables=['chain1', 'chain2'])

merged_chain = prompt3 | model | parser

chain = parellael_chain | merged_chain

result = chain.invoke({
    'content':content
})

print(result)