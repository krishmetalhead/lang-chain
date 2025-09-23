from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path='dir', glob='*', loader_cls=PyPDFLoader)

files = loader.load()

for  i in files :
    print(i.page_content)

