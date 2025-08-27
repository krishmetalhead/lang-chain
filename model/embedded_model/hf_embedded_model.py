from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
#from numpy import np
from dotenv import load_dotenv

load_dotenv()
hf_end_point =HuggingFaceEndpointEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2')
documents = [
    'Adriraj is a nice boy, who loves to play Video games',
    'Krishnendu lives in Kolkata, who works in capgemini', 
    'Sayanti lives in Dhakuria and loves online shopping'
]
hf_emmbedded_documents = hf_end_point.embed_documents(documents)
hf_embedded_query = hf_end_point.embed_query("Tell me about Sayanti")
scores = (cosine_similarity([hf_embedded_query],hf_emmbedded_documents)[0])
sorted_emdeddings = sorted(list(enumerate(scores)) , key= lambda x : x[1])
#print(sorted_emdeddings)

index , score  = sorted_emdeddings[-1]
print(documents[index])



