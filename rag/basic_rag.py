from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is one of the greatest batsman , who plays for RCB in IPL", 
    metadata={'team':'RCB'}    
)

doc2 = Document(
    page_content="Jasprit Bumrah is one of the greatest bowler , who plays for MI in IPL", 
    metadata={'team':'MI'}    
)

doc2 = Document(
    page_content="Christinao Ronaldo , is one of the GOAT in football. Currently playing for Al Nasar", 
    metadata={'team':'Al Nasar'}    
)

embeddings = HuggingFaceEmbeddings()
docs = [doc1,doc2]
print('start adding in chroma')
vector_stote = Chroma(
    embedding_function=embeddings,
    persist_directory='chroma_db',
    collection_name='sample'
)
vector_stote.add_documents(docs)
print('end adding in chroma')
data = vector_stote.get(include=['embeddings'])
print(data)
result = vector_stote.similarity_search(
    query='Who plays for the biggest club from Saudi Arabia',k=1
)
print(result)

