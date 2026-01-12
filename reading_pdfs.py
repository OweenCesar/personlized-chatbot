# reading pdfs , preparıng embeddings 

from langchain_community.document_loaders import PyPDFLoader

cv_path = "resume_oween_.pdf"
 

def get_pdf_texts(cv_path):
    loader = PyPDFLoader(cv_path)
    docs = loader.load()
    texts = [doc.page_content for doc in docs]
    return texts

if __name__ == "__main__":
    texts = get_pdf_texts(cv_path)
    print(texts)