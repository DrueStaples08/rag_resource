# Import Packages
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.chat_models import ChatOpenAI
from langserve import add_routes
from fastapi import FastAPI
import uvicorn
import os
from dotenv import load_dotenv

# Load in environment variable 
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Create API
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="API Server Demo"
)

# # Add routes example
# add_routes(
#     app,
#     Ollama(),
#     path="/ollama"
# )

model_llama2 = Ollama(model="llama2")
model_gemma = Ollama(model="gemma2")

# prompt1 = ChatPromptTemplate.from_messages(
#     [
#         # ("Explain how a {topic} is made."),
#         # ("What is a {topic} most threatening train"),
#         ("Rearrange the letters of the input text {topic} to create a first, middle, and last name.")
#         ]
#     )

prompt1 = ChatPromptTemplate.from_messages(
    ["Write me a love letter about {topic} under 10 words"]
)

prompt2 = ChatPromptTemplate.from_messages(
    ["Write me a haiku about {topic} please"]
)


# Add Routes
add_routes(
    app,
    prompt1|model_llama2,
    path="/llama2_response"
)

add_routes(
    app, 
    prompt2|model_gemma,
    path="/gemma_response"
)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)





