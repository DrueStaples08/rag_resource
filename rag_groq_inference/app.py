import os
from dotenv import load_dotenv
# Ollama embeddings
from langchain_community.embeddings import OllamaEmbeddings
# Document Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Combine Documents Chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Load Web Data
from langchain_community.document_loaders import WebBaseLoader
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

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loader = WebBaseLoader("https://footballdatabase.com/ranking/north-america/1")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    


st.title('Langchain with Groq')
llm = ChatGroq(
    groq_api_key=groq_api_key,
    # model_name = "Llama3-8b-8192"
    # model_name = "Gemma-7b-It"
    model_name = "mixtral-8x7b-32768"

    )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Questions:{input}
    """

    )

docs_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()

retrieval_chain = create_retrieval_chain(retriever, docs_chain)

prompt = st.text_input("Input your prompt: ")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("Response Time Length: ", time.process_time()-start_time)
    st.write(response['answer'])




