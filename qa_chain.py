from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_classic.chains import retrieval_qa,RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()








def create_qa_chain(vector_db):

    retrieval_qa =vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    llm = ChatGroq(
        model ="llama-3.1-8b-instant",
        temperature =0.7,
    )

    prompt =ChatPromptTemplate.from_template(
        """You are a helpful assistant. 
        Use the following retrieved chunks {context} to answer the question{input}.
        If you don't know the answer, say you don't know."""
    )

    documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    qa_chain = create_retrieval_chain(
        retriever=retrieval_qa, 
        combine_documents_chain=documents_chain
   )
    
    return qa_chain


            
        

        

        
            
        


