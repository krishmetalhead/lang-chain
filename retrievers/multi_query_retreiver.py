#It addresses the challenge of formulating the ideal query string for information retrieval by generating multiple variations of a user's initial query.

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

doc1 = Document(page_content="Drinking 5 ltr of water daily")
doc2 = Document(page_content="Spend 30 min in execise on daily basis")
doc3 = Document(page_content="Avoid smoking and drinking ")
doc4 = Document(page_content="Avoid wasting water to save the earrth from drying")
doc5 = Document(page_content="Try to indulge in some hobby to avoid stress")
doc6 = Document(page_content="Nurture rain water harvesting ")
doc7 = Document(page_content="Use EV cars for healthier ecosystem which does lesser carbon emission ")

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=huggingfaceEndpointObject)

embeddings = HuggingFaceEmbeddings()
docs = [doc1,doc2,doc3,doc4,doc5,doc6,doc7]
print('start adding in chroma')
vector_stote = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)
vector_stote.add_documents(docs)
print('end adding in chroma')

multiQueryRetreiver = MultiQueryRetriever.from_llm(
    llm=model,
    retriever=vector_stote.as_retriever(search_kwargs={"k":3})
)

query = "How to save the earth"

retrieved_docs = multiQueryRetreiver.invoke(query)

for doc in retrieved_docs:
    print(doc.page_content)


