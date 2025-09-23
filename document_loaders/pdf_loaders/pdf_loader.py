from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='documnent2.pdf')

files = loader.load()

for  i in files :
    print(i.page_content)
