from langchain_community.vectorstores import FAISS
def create_FAISS_vectorstore (chunks,embeddings):
    vectordb=FAISS.from_documents(document=chunks,embedding=embeddings)
    vectordb.save_local("faiss_store")
    return vectordb

