from langchain_community.vectorstores import Chroma


def create_chroma_vectorstore(chunks, embeddings):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstores"
    )
    vectordb.persist()
    return vectordb






