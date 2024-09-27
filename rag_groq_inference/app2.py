import os 
# UI for LLM
import streamlit as st 
# Fast inference for Chat API
from langchain_groq import ChatGroq
# Loads data from the web
from langchain_community.document_loaders import WebBaseLoader
# Open Source Embeddings
from langchain_community.embeddings import OllamaEmbeddings
# Open Source Models
from langchain_community.llms import Ollama
# Chunk Data
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Creates a chain (Combine) for passing all documents to llm
from langchain.chains.combine_documents import create_stuff_documents_chain
# Template for Chat model prompts (i.e. additional instructions as a str type to enhance the llm)
from langchain.prompts import ChatPromptTemplate
# Retriever is an UI whose main purpose is fast loading/retrieval of documents to/from a vector store
"""
create_retrieval_chain vs create_retrival_tool

create_retrieval_chain: build complex retrieval processes involving multiple steps
create_retrival_tool: simple retrieval tool mechanism
"""
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.retrieval import create_retrieval_chain
# Vector Database. Other examples include ChromaDB and Lance
from langchain_community.vectorstores import FAISS
# Load in environment variables (from .env file) 
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]

# Load the web data
web_data = WebBaseLoader("https://footballdatabase.com/ranking/north-america/1")
web_docs = web_data.load()

# Chunk Documents with text splitter
docs_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(web_docs)

# Generate Prompt
prompt = ChatPromptTemplate({
    """
    Keep track of the team rankings and 
    how they compared to last year which 
    is also included in the HTML. 
    <context>
    {context}
    </context>
    Input: {input}
    """
})

# Create model
llm = Ollama(model="llama2")

# Create Chain to contain all document chunks
docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

# Create Database to store the embedding
db = FAISS.from_documents(documents=docs_split, embedding=OllamaEmbeddings())

# Create retriever 
retriever = db.as_retriever()

# Create the retrievel tool 
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name='soccer_ranker',
    description="Contains the current top 50 North American soccer clubs with their rankings."
)

# Where to put my docs_chain, and retrieval_tool
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=docs_chain
)

# user_query_result = retrieval_chain.invoke({"input": "What is the highest ranked soccer team in the United States?"})
user_query_result = retrieval_chain.invoke({"input": "What MLS team has improved the most since last year?"})


if __name__ == "__main__":
    print(user_query_result["answer"])
    """
    To determine which MLS team has improved the most since last year, we can compare the current ranking of each team in the HTML provided to their ranking from last year.

    Here are the top 5 MLS teams that have improved the most since last year:

    1. CF Pachuca (Mexico) - Current ranking: 1st, Last year's ranking: 3rd (-2 spots)
    2. Tigres UANL (Mexico) - Current ranking: 2nd, Last year's ranking: 5th (-3 spots)
    3. Monterrey (Mexico) - Current ranking: 3rd, Last year's ranking: 7th (-4 spots)
    4. Guadalajara (Mexico) - Current ranking: 4th, Last year's ranking: 9th (-5 spots)
    5. Seattle Sounders FC (United States) - Current ranking: 5th, Last year's ranking: 12th (+7 spots)

    Note that the ranking is based on the teams' performance in the CONCACAF region, which includes MLS and other teams from North America, Central America, and the Caribbean. 
    """




