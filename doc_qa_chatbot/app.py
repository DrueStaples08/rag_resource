import os
from dotenv import load_dotenv
# Ollama embeddings
from langchain_community.embeddings import OllamaEmbeddings
# Document Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Combine Documents Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Load Web Data
from langchain_community.document_loaders import PyPDFDirectoryLoader
# Create a retrieval chain for more complex chains (as opposed to create_retrieval_tool)
from langchain.chains import create_retrieval_chain
# UI for interactive data apps
import streamlit as st
# Utilize LPU (Language Processsing Unit) for fast inference
# Wrapper for the Groq API that uses llms finetuned for chat
from langchain_groq import ChatGroq
# Prompt templates
from langchain_core.prompts import ChatPromptTemplate
# Vector Store
from langchain_community.vectorstores import FAISS
# Keep track of the time duration of script 
import time


load_dotenv()

# Load Groq API Key
groq_api_key = os.environ["GROQ_API_KEY"]

llm = ChatGroq(groq_api_key=groq_api_key)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions using the context from the input PDF files. 
        Give only a 1 sentence answer with at 3-5 bullet points of facts to prove your point. 
        <context>{context}<context>
        Question: {input}
    """)





def streamlit_variables():
    # vector or vectors? should be vectors 
    # if "vector" not in st.session_state:
    # if "vectors" not in st.session_state:
    if "db" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.docs_loader = PyPDFDirectoryLoader(path="./pdf_sources")
        # print('A')
        # print(st.session_state.docs_loader)
        # print("\n\n")
        # Must be inside doc_qa_chatbot/ folder when running script or this line below will cause an error
        st.session_state.docs = st.session_state.docs_loader.load()[:5]
        # print('B')
        # print(st.session_state.docs, len(st.session_state.docs))
        # print("\n\n")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # print('C')
        # print(st.session_state.text_splitter)
        # print("\n\n")
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        # print('D')
        # print(st.session_state.final_documents)
        # print("\n\n")
        st.session_state.db = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        # print('E')
        # print(st.session_state.vectors)
        # print("\n\n")
    return st.session_state.db

# Streamlit input question
streamlit_prompt = st.text_input("Enter Question on RAG Systems or LLM's")

def chain_prompt(llm, prompt, streamlit_prompt):
    # 1. The way this workflow operates, it first needs verification that there is indeed an input question
    if streamlit_prompt:
        docs_chain = create_stuff_documents_chain(llm, prompt)
        # 2. Next, the streamlit variables can be instatiated.
        # 2a. If the database already exists, then so do the other variables - therefore don't recreate them
        # ...unless there's plans to utilize other embedding models, databases, etc.
        # 2b. If the database doesn't exist, load in the data from the source and then create the steamlit variables for RAG pipelines
        db = streamlit_variables()
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, docs_chain)
        return retrieval_chain

 

def streamlit_main_run(llm, prompt, streamlit_prompt):
    if st.button("Answer Question"):
        st.write('Start....')
        retrieval_chain = chain_prompt(llm, prompt, streamlit_prompt)
        # 3. Finally, invoke the chain to perform question-answering - ask a question based on the data source - recieve a compitent and sensical answer
        response = retrieval_chain.invoke({"input": streamlit_prompt})
        st.write(response["answer"])
        st.write("End of Reponse")


if __name__ == "__main__":
    streamlit_main_run(llm, prompt, streamlit_prompt)





