# Tutorial  2: Use Fast API and LangServe to deploy Langchain application.

## Resource: 
(https://www.youtube.com/watch?v=XWB5DXP-DO8&list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ&index=6)

## Files: 
    - app.py : FastAPI with Swagger UI
    - client.py : make post requests to Streamlit UI

## FastAPI: fast way to create web apps 

## LangServe: deployment of Langchain apps

## Note: Swagger is already provided by Langchain, so you don't need to config a yaml file for it.

## Design:
    App (Web or Mobile) ---> API ---> Route ---> LLMs (OpenAI, Ollama, etc)

## Download Open Source Models:
    To download a model, run "ollama run [model_name]" in terminal, e.g. "ollama run llama2" 
    (Resource: https://ollama.com/library)

## 1. Run Application (app.py)
- To run application, run "python app.py"
- Click on http://localhost:8000/
- type in "/docs" after the given webaddress i.e. http://localhost:8000/docs

## 2. Test the endpoints directly (terminal):
- Keep Fast API running from the step above.
- Open another terminal (make sure to source ~/.zshrc and activate pyenv vm)
- Run the following commands:
    - curl -X POST "http://localhost:8000/llama2_response/invoke" -H "Content-Type: application/json" -d '{"input": {"topic": "test"}}'
    - curl -X POST "http://localhost:8000/gemma_response/invoke" -H "Content-Type: application/json" -d '{"input": {"topic": "test"}}'

## 3. Run API (client.py)
- Run Streamlit -> "streamlit run client.py"
