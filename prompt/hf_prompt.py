#https://www.youtube.com/watch?v=3TGqlQxpuU0
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

dynamic_prompt = 'India' #should come from UI


prompt = PromptTemplate.from_template("Which footbal club is the most successful in {foo}")
#prompt.format(foo="Europe")
prompt_val = prompt.invoke({
    'foo' : dynamic_prompt
})
model = ChatHuggingFace(llm=huggingfaceEndpointObject)
result = model.invoke(prompt_val)
print(result.content) #should be the response back to the user