from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_classic.chains import retrieval_qa,RetrievalQA




def create_qa_chain(vector_db):

    retrieval_qa =vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    LLM= ChatGroq(
        model="llama-3.1-8b-instant",
        temperature =0.7,
    )
    qa_chain = RetrievalQA(LLM=LLM,retriever=retrieval_qa)
    return qa_chain


         
    

    

    
        
    

)
