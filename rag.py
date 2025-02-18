from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import warnings
import numpy as np
import faiss
from dotenv import load_dotenv

# Import LangChain components
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document  # Fixed import

# Environment and warning setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Global variable to cache the vector store
VECTOR_STORE = None

# Pydantic models for request validation
class Tool(BaseModel):
    name: str
    description: str
    api_details: str

class QueryRequest(BaseModel):
    tool_data: Optional[List[Tool]] = None
    user_query: Optional[str] = None

def convert_tools_to_documents(tool_data):
    """
    Converts tool data into Document objects.
    """
    documents = []
    for tool in tool_data:
        content = (
            f"Name: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"API Details: {tool.api_details}"
        )
        documents.append(Document(page_content=content))
    return documents

def format_docs(docs):
    """
    Concatenates a list of Document objects into a single string.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def split_document(document, chunk_size=200, overlap=50):
    """
    Splits a Document into smaller chunks if its content exceeds chunk_size.
    Returns a list of Document objects.
    """
    content = document.page_content
    if len(content) <= chunk_size:
        return [document]
    chunks = []
    start = 0
    while start < len(content):
        chunk = content[start:start+chunk_size]
        chunks.append(Document(page_content=chunk))
        start += chunk_size - overlap
    return chunks

def setup_vector_store(documents):
    """
    Creates a FAISS vector store from a list of documents (after splitting them into chunks)
    using an Ollama embedding model.
    """
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    
    # Use the simpler from_documents constructor
    all_chunks = []
    for doc in documents:
        chunks = split_document(doc)
        all_chunks.extend(chunks)
        
    vector_store = FAISS.from_documents(documents=all_chunks, embedding=embeddings)
    return vector_store

def create_rag_chain(retriever):
    """
    Creates a RAG chain with an optimized prompt template and ChatOllama model.
    """
    prompt = """
You are an expert AI assistant specializing in selecting the best matching tool from a given list based on user requirements. Analyze the user query carefully and match it with the provided tool information. Consider the tool name, description, and API details to determine the top three most relevant tools. Provide your answer in clear bullet points with detailed reasoning. If no tool is highly relevant, explicitly state that no matching tool was found.

Your answer must follow this format:
- **Tool Name**: [Name]
  - *Description*: [Brief description and why it matches]
  - *API Details*: [Relevant API info]

Question: {question}

Tool Information: {context}
    """
    model = ChatOllama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        top_p=0.9,
        min_tokens=50,
        temperature=0.3,
        presence_penalty=0.0,
        stream=False,  # Changed to False for API use
        max_tokens=500,
        system="You are a highly accurate tool selection assistant who provides concise, fact-based recommendations with expert reasoning."
    )
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return rag_chain

def get_vector_store(tool_data=None):
    """
    Returns the global VECTOR_STORE. If tool_data is provided (non-empty list),
    it rebuilds the vector store with the new data. Otherwise, it returns the cached store.
    """
    global VECTOR_STORE
    if tool_data and len(tool_data) > 0:
        # Build new vector store from the provided tool_data
        documents = convert_tools_to_documents(tool_data)
        VECTOR_STORE = setup_vector_store(documents)
    elif VECTOR_STORE is None:
        raise ValueError("No vector store available. Provide tool_data at least once.")
    return VECTOR_STORE

def run_rag_system(user_query, tool_data=None):
    """
    Runs the RAG system: If tool_data is provided, it rebuilds the vector store.
    Otherwise, it uses the cached vector store to search and answer the query.
    """
    vector_store = get_vector_store(tool_data)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    rag_chain = create_rag_chain(retriever)
    response = rag_chain.invoke(user_query)
    return response

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG Tool Recommendation API",
    description="API for recommending AI tools based on user queries using RAG",
    version="1.0.0"
)

@app.post("/rag/")
async def rag_endpoint(request: QueryRequest):
    """
    Endpoint that handles three scenarios:
    1. If both tool_data and user_query are provided: Add tools to vector store and process query
    2. If only tool_data is provided: Add tools to vector store and return success message
    3. If only user_query is provided: Process query using existing vector store
    """
    try:
        # Case 1: Both tool_data and user_query are provided
        if request.tool_data and request.user_query:
            response = run_rag_system(request.user_query, request.tool_data)
            return {
                "success": True,
                "message": "Tools added to database and query processed",
                "response": response
            }
            
        # Case 2: Only tool_data is provided
        elif request.tool_data:
            get_vector_store(request.tool_data)
            return {
                "success": True,
                "message": f"Added {len(request.tool_data)} tools to the database"
            }
            
        # Case 3: Only user_query is provided
        elif request.user_query:
            response = run_rag_system(request.user_query)
            return {
                "success": True,
                "message": "Query processed using existing database",
                "response": response
            }
            
        # Error case: Neither tool_data nor user_query is provided
        else:
            raise HTTPException(
                status_code=400,
                detail="Either tool_data or user_query (or both) must be provided"
            )
            
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional
# import uvicorn
# import os
# import warnings
# import numpy as np
# import faiss
# from dotenv import load_dotenv
# import json
# import requests
# import logging
# import time
# from datetime import datetime

# # Import LangChain components
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.docstore.document import Document

# # Setup logging configuration
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('rag_system.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# # Environment and warning setup
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# warnings.filterwarnings("ignore")
# load_dotenv()

# # Constants
# API_URL = "http://91.150.160.38:1365/api"
# VECTOR_STORE = None

# # Pydantic models for request validation
# class Tool(BaseModel):
#     name: str
#     description: str
#     api_details: str

# class QueryRequest(BaseModel):
#     tool_data: Optional[List[Tool]] = None
#     user_query: str

# class CustomOllamaEmbeddings(OllamaEmbeddings):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.base_url = API_URL
#         logger.info("Initialized CustomOllamaEmbeddings with base_url: %s", self.base_url)

#     def _embed_function(self, text):
#         logger.debug("Generating embedding for text: %s", text[:100] + "..." if len(text) > 100 else text)
#         try:
#             response = requests.post(
#                 f"{self.base_url}/generate",
#                 json={
#                     "model": "nomic-embed-text",
#                     "prompt": text,
#                     "stream": False
#                 }
#             )
#             result = response.json()
#             logger.debug("Successfully generated embedding")
#             return result.get("embedding", [])
#         except Exception as e:
#             logger.error("Error generating embedding: %s", str(e))
#             raise

# class CustomChatOllama(ChatOllama):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.base_url = API_URL
#         logger.info("Initialized CustomChatOllama with base_url: %s", self.base_url)

#     def _generate_response(self, prompt):
#         logger.debug("Generating response for prompt: %s", prompt[:100] + "..." if len(prompt) > 100 else prompt)
#         try:
#             response = requests.post(
#                 f"{self.base_url}/generate",
#                 json={
#                     "model": self.model,
#                     "prompt": prompt,
#                     "stream": False
#                 }
#             )
#             result = response.json()
#             logger.debug("Successfully generated response")
#             return result.get("response", "")
#         except Exception as e:
#             logger.error("Error generating response: %s", str(e))
#             raise

# def convert_tools_to_documents(tool_data):
#     """Converts tool data into Document objects."""
#     logger.info("Converting %d tools to documents", len(tool_data) if tool_data else 0)
#     documents = []
#     for i, tool in enumerate(tool_data):
#         content = (
#             f"Name: {tool.name}\n"
#             f"Description: {tool.description}\n"
#             f"API Details: {tool.api_details}"
#         )
#         documents.append(Document(page_content=content))
#         logger.debug("Converted tool %d: %s", i + 1, tool.name)
#     return documents

# def format_docs(docs):
#     """Concatenates Document objects into a single string."""
#     logger.debug("Formatting %d documents", len(docs))
#     return "\n\n".join([doc.page_content for doc in docs])

# def split_document(document, chunk_size=200, overlap=50):
#     """Splits a Document into smaller chunks."""
#     logger.debug("Splitting document with chunk_size=%d, overlap=%d", chunk_size, overlap)
#     content = document.page_content
#     if len(content) <= chunk_size:
#         logger.debug("Document smaller than chunk size, returning as is")
#         return [document]
#     chunks = []
#     start = 0
#     while start < len(content):
#         chunk = content[start:start+chunk_size]
#         chunks.append(Document(page_content=chunk))
#         start += chunk_size - overlap
#     logger.debug("Split document into %d chunks", len(chunks))
#     return chunks

# def setup_vector_store(documents):
#     """Creates a FAISS vector store using custom embeddings."""
#     logger.info("Setting up vector store with %d documents", len(documents))
#     start_time = time.time()
    
#     try:
#         embeddings = CustomOllamaEmbeddings(model='nomic-embed-text')
#         logger.debug("Generating sample vector for initialization")
#         sample_vector = embeddings.embed_query("sample text")
#         index = faiss.IndexFlatL2(len(sample_vector))
        
#         vector_store = FAISS(
#             embedding_function=embeddings,
#             index=index,
#             docstore=InMemoryDocstore(),
#             index_to_docstore_id={}
#         )
        
#         logger.debug("Processing document chunks")
#         all_chunks = []
#         for doc in documents:
#             chunks = split_document(doc)
#             all_chunks.extend(chunks)
            
#         vector_store.add_documents(documents=all_chunks)
        
#         setup_time = time.time() - start_time
#         logger.info("Vector store setup completed in %.2f seconds", setup_time)
#         return vector_store
#     except Exception as e:
#         logger.error("Error setting up vector store: %s", str(e))
#         raise

# def create_rag_chain(retriever):
#     """Creates a RAG chain with custom chat model."""
#     logger.info("Creating RAG chain")
#     prompt = """
#     You are an assistant that selects the best matching AI tool based on user requirements.
#     Use the following tool information to answer the query.
#     If no tool is relevant, state that no matching tool was found.

#     List Top 3 search results within my data only.
    
#     Question: {question}
#     Tool Information: {context}
    
#     Answer in bullet points:
#     """
#     try:
#         model = CustomChatOllama(model="deepseek-r1:7b")
#         prompt_template = ChatPromptTemplate.from_template(prompt)
        
#         rag_chain = (
#             {"context": retriever | format_docs, "question": RunnablePassthrough()}
#             | prompt_template
#             | model
#             | StrOutputParser()
#         )
#         logger.info("RAG chain created successfully")
#         return rag_chain
#     except Exception as e:
#         logger.error("Error creating RAG chain: %s", str(e))
#         raise

# def get_vector_store(tool_data=None):
#     """Manages the vector store cache."""
#     global VECTOR_STORE
#     logger.info("Getting vector store (tool_data provided: %s)", "yes" if tool_data else "no")
    
#     if tool_data and len(tool_data) > 0:
#         logger.info("Creating new vector store with %d tools", len(tool_data))
#         documents = convert_tools_to_documents(tool_data)
#         VECTOR_STORE = setup_vector_store(documents)
#     elif VECTOR_STORE is None:
#         logger.error("No vector store available and no tool_data provided")
#         raise ValueError("No vector store available. Provide tool_data at least once.")
    
#     return VECTOR_STORE

# def run_rag_system(user_query, tool_data=None):
#     """Runs the RAG system with the new API."""
#     start_time = time.time()
#     request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
#     logger.info("Starting RAG system for request %s", request_id)
#     logger.info("Query: %s", user_query)
    
#     try:
#         logger.info("Step 1/3: Initializing vector store")
#         vector_store = get_vector_store(tool_data)
        
#         logger.info("Step 2/3: Creating retriever")
#         retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
        
#         logger.info("Step 3/3: Creating and running RAG chain")
#         rag_chain = create_rag_chain(retriever)
#         response = rag_chain.invoke(user_query)
        
#         process_time = time.time() - start_time
#         logger.info("Request %s completed in %.2f seconds", request_id, process_time)
#         return response
#     except Exception as e:
#         logger.error("Error processing request %s: %s", request_id, str(e), exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# # Initialize FastAPI app
# app = FastAPI(
#     title="RAG Tool Recommendation API",
#     description="API for recommending AI tools based on user queries using RAG",
#     version="1.0.0"
# )

# @app.post("/recommend/")
# async def recommend_tools(request: QueryRequest):
#     """Endpoint to recommend tools based on user query and optional tool data."""
#     start_time = time.time()
#     request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
#     logger.info("Received request %s", request_id)
#     logger.info("Query: %s", request.user_query)
#     logger.info("Number of tools provided: %d", len(request.tool_data) if request.tool_data else 0)
    
#     try:
#         response = run_rag_system(
#             user_query=request.user_query,
#             tool_data=request.tool_data
#         )
#         process_time = time.time() - start_time
#         logger.info("Request %s completed successfully in %.2f seconds", request_id, process_time)
#         return {
#             "request_id": request_id,
#             "response": response,
#             "processing_time": f"{process_time:.2f}s"
#         }
#     except ValueError as ve:
#         logger.error("Validation error for request %s: %s", request_id, str(ve))
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         logger.error("Internal error for request %s: %s", request_id, str(e), exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     logger.info("Starting RAG Tool Recommendation API server")
#     uvicorn.run(app, host="0.0.0.0", port=8000)