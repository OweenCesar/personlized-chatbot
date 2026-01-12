import requests
# to load pdfs and prepare texts
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Embeddings and vector store

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


cv_path = "resume_oween_.pdf"

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "deepseek-r1:1.5b",
        "stream": False, # get one single json response 
        "messages": [{"role": "user", "content": "hey where are you from actually?"}],
    },
    timeout=120,
)
print(resp.json()["message"]["content"])


# UNO 
def get_pdf_texts(cv_path):
    # getting our data from the pdfs, returns 'texts' 
    loader = PyPDFLoader(cv_path)
    docs = loader.load()
     
    return docs



# DOS
# what does RecursiveCharacterTextSplitter do?
def split_texts(texts):
    # splitting texts into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_texts = []
    for text in texts:
        splits = text_splitter.split_text(text)
        split_texts.extend(splits)
    return split_texts




# Consolidation of the vector store function

def create_vector_store(path_to_pdf):
    document = get_pdf_texts(path_to_pdf)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]) # separators refers to prıority of splittıng \n\n means splitting by paragraphs
    split_docs = text_splitter.split_documents(document)

    embeddings = OllamaEmbeddings(
    model="all-minilm:l6-v2",
    base_url="http://localhost:11434",
    )

    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store

# Example usage

create_vector_store("resume_oween_.pdf")





 