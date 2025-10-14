#Compute Maximal Marginal Relevance (MMR). MMR is a technique used to select documents that are both relevant to the query and diverse among themselves
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

doc1 = Document(page_content="Virat Kohli plays for Team India")
doc2 = Document(page_content="Virat Kohli is the key player and former captain of RCB in IPL")
doc3 = Document(page_content="Virat Kohli plays for RCB ")
doc4 = Document(page_content="Virat Kohli is married to Anoushka Sharma ")
doc5 = Document(page_content="ISL is tier 1 football tournament of India ")
doc6 = Document(page_content="Capgemini is Frenchh IT company ")

embeddings = HuggingFaceEmbeddings()
docs = [doc1,doc2,doc3,doc4,doc5,doc6]
print('start adding in chroma')
vector_stote = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)
vector_stote.add_documents(docs)
print('end adding in chroma')

retriever = vector_stote.as_retriever(
     search_type="mmr",
     search_kwargs={"k":3, "lambda_mult":0.5}
)
#k = How many documents to be fetched
#lambda_mult = Degree of removal of similar redundant documents . If 1 its same as similarity search

query = "Tell me about Virat Kohli"
retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(doc.page_content)

