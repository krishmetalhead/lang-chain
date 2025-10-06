from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
I am Krishnendu Bhowmick

I love to watch Football

Bayern Munich is my favourite team.
They won UCL 5 times.
They won the Bundesliga title most number of times
"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)
result = text_splitter.split_text(text)
print(result)