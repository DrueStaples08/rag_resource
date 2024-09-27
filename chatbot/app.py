# pay-to-use
# from langchain_openai import ChatOpenAI
# open source
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# For LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        # The question asked
        ("user", "Question:{question}")
    ]
)

# streamlit framework
st.title("Langchain with OPENAI model API")
input_text=st.text_input("Search the topic u want. ")

# OpenAI LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo")
# Ollama
llm = Ollama(model="llama2")

'''
Download models from ollama (open sourced pre-trained models)

ollama run [model_name]
to download specific model
https://ollama.com/library
'''
output_parser = StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))


'''
Location: http://localhost:8501/

Input: hey hi

Output:
RateLimitError: Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}

Run app through streamlit in terminal:
streamlit run app.py
'''



