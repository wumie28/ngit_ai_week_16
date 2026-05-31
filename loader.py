from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,UnstructuredExcelLoader
import os



def load_document(filepath):
    loader = PyPDFLoader(filepath)
    return loader.load()


   