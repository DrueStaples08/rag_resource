# Tutorial 3: How to Create a RAG Pipeline with Langchain, ChromaDB, and FAISS

## RAG 
- Retrieval Augmented Generation, pipeline to save embeddings, and to perform quick retrieval for user requests. (E.g. user produces a question, and receives and answer.)

## Vector Stores

- Database specifically designed for vector embeddings
- e.g. Pinecone, ChromaDB

### Vector Store Architecture
- Data Source(s) (e.g. csv, pdf, txt, images, etc. ),(Load, Transform, and Embed Data) -> vector database
- User query -> LLM (transforms query into embeddings) -> vector store -> Output


## Chunking
For large amounts of data, chunking can help with faster computational processing with less memory. This can be anything, from chunking chapters, pages, paragraphs, sentences, etc. 

Chunks:
- Chunk 0: Artificial Intelligence (AI) is a rapidly growing field.
- Chunk 1: It has applications in various domains, including healthcare, finance, and transportation.
- Chunk 2: Machine Learning (ML) is a subset of AI.
- Chunk 3: It focuses on building systems that can learn from data.
- Chunk 4: Deep Learning (DL) is a subset of ML that uses neural networks with many layers.


## Indexing
Every embedding has a key like a dictionary. Indexing means to effectively retrieve the correct data based on the words in the input query. There are multiple ways to index. One example being the index is a dictionary that contains the words in corpus as keys and a list of chunk indices as values that represents where each chunk the word is located. For something like NER applications, it is considered useful to index the metadata. The metadata is just a list of elements that relate to the key.


Index:
- artificial: [0]
- intelligence: [0]
- ai: [0]
- is: [0, 2, 4]
- a: [0, 2, 4]
- rapidly: [0]
- growing: [0]
- field: [0]
- it: [1, 3]
- has: [1]
- applications: [1]


Indexing metadata: e.g. [id, entity]
sequence = "Can I get 100 10 foot 2x4's of oak delivered saturday the 12th at 9am. 

'100' [102, 'inventory_count'] 
'10 foot' [242, 'dimensions']
'2x4's' [40, 'product']
'oak' [65, 'material']
saturday [392, 'day_of_week']
12th [533, 'day_of_month']
9am [346, 'time']




