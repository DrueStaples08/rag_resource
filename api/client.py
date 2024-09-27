import requests
import streamlit as st

def get_llama2_response(input_text:str):
    response=requests.post("http://localhost:8000/llama2_response/invoke",
    # http://localhost:8000/docs#/default/create_feedback_from_token_llama2_response_token_feedback_post
        json = {
            "input": {"topic": input_text}
        }
    )
    # st.write("LLama JSON Response", response.json()["output"])
    return response.json()["output"]

def get_gemma_response(input_text:str):
    response = requests.post("http://localhost:8000/gemma_response/invoke",
        json = {
            "input": {"topic": input_text}
            }
    )
    # st.write("Gemma JSON Response", response.json())
    return response.json()["output"]

# streamlit framework
st.title("Haiku and Love Letter")
input_text=st.text_input("Jot a love letter for")
input_text2=st.text_input("Write a haiku about")

if input_text:
    st.write(get_llama2_response(input_text))

if input_text2:
    st.write(get_gemma_response(input_text2))



