from langchain.text_splitter import CharacterTextSplitter

text = 'Hello World'
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0,separator='')
result = text_splitter.split_text(text)
print(result)