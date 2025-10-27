from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import Document
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

doc1 = Document(page_content=("""
                Durga Puja is biggest Bengali festival
                IID happens twice a year
                Durga godess kills Mahisasura
                """), metadata={"source":"Doc1"})
doc2 = Document(page_content=("""
                Durga Puja occures in Autum
                Chrsistmas is biggest festival for Christians
                Jains worships Mahavir
                """), metadata={"source":"Doc2"})

huggingfaceEndpointObject = HuggingFaceEndpoint(
    repo_id= "meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=huggingfaceEndpointObject)

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

baseQueryRetreiver = vector_stote.as_retriever(search_kwargs={"k":1})
compression = LLMChainExtractor.from_llm(model)

contextQueryRetreiver = ContextualCompressionRetriever(
    base_retriever=baseQueryRetreiver,base_compressor=compression)

compressed_results = contextQueryRetreiver.invoke("Define Durga Puja")

for i , doc in enumerate(compressed_results):
    print(doc.page_content)