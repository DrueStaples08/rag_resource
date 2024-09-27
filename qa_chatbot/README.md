# Tutorial 4: QA Chatbot with Langchain's Chains and Retrievers

RAG
1. Ingest Data (text or pdf files, from the web)
2. Chunk and generate data embeddings with LLM
3. Load into vector store (ChromaDB, FAISS, Lance, Pinecone)

4. User makes a query
5. Query is fed into LLM to generate the query embeddings
6. The embeddings are then used to help retrieve the closet K-nearest neighbors, i.e the most similar results are selected 


Prompts -> Chain and Retriever (LLM)-> Response

Stuff Document Chain

Resources: 
- Tutorial Advanced RAG [https://www.youtube.com/watch?v=tIwi92nkcu0&list=PLZoTAELRMXVOQPRG7VAuHL--y97opD5GQ&index=6]
- LCEL Chains (LangChain Expressed Language) [https://python.langchain.com/v0.1/docs/modules/chains/]
