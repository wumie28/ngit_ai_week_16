import streamlit as st
import os
import tempfile

from loader import load_document
from qa_chain import create_qa_chain
from splitters import split_documents
from embeddings import get_embeddings
from faiss_vectorstore import create_FAISS_vectorstore
from chroma import create_chroma_vectorstore

st.set_page_config(
    page_title="RAG QA Application",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("RAG QA APPLICATION")

vector_option = st.radio(
    "Select Vector Database",
    ("Chroma", "FAISS")
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "docx"]
)

if uploaded_file is not None:

    file_extension = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
        ) as temp_file:

        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    with st.spinner("Processing document..."):

        document = load_document(temp_path)

        chunks = split_documents(document)

        embeddings = get_embeddings()

        if vector_option == "FAISS":
            vectorstore = create_FAISS_vectorstore(
                chunks,
                embeddings
            )
        elif vector_option == "Chroma":
            vectorstore = create_chroma_vectorstore(
                chunks,
                embeddings
            )

        qa_chain = create_qa_chain(vectorstore)

        st.success("Document processed successfully!")

    query = st.text_input("Ask a question")

    if query:

        # docs = vectorstore.similarity_search(
        #     query,
        #     k=3
        # )

        #st.subheader("Retrieved Chunks")

        #for doc in docs:
            #st.write(doc.page_content)
            #st.write("---")
        response = qa_chain.invoke({"input": query})
        st.write(response.get("answer"))   
    os.remove(temp_path) 
    




  