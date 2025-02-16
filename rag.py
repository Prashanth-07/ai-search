#!/usr/bin/env python
import os
import warnings
import numpy as np
import faiss
from dotenv import load_dotenv
import json
import requests

# Import LangChain components
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document

# Environment and warning setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Constants
API_URL = "http://91.150.160.38:1365/api"
VECTOR_STORE = None

class CustomOllamaEmbeddings(OllamaEmbeddings):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = API_URL

    def _embed_function(self, text):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": "nomic-embed-text",
                "prompt": text,
                "stream": False
            }
        )
        result = response.json()
        # Parse the embedding from the response
        # You'll need to adjust this based on the actual response format
        return result.get("embedding", [])

class CustomChatOllama(ChatOllama):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = API_URL

    def _generate_response(self, prompt):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        return result.get("response", "")

def convert_tools_to_documents(tool_data):
    """
    Converts tool data into Document objects.
    """
    documents = []
    for tool in tool_data:
        content = (
            f"Name: {tool.get('name', 'N/A')}\n"
            f"Description: {tool.get('description', 'No description provided.')}\n"
            f"API Details: {tool.get('api_details', 'N/A')}"
        )
        documents.append(Document(page_content=content))
    return documents

def format_docs(docs):
    """
    Concatenates Document objects into a single string.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def split_document(document, chunk_size=200, overlap=50):
    """
    Splits a Document into smaller chunks.
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
    Creates a FAISS vector store using custom embeddings.
    """
    embeddings = CustomOllamaEmbeddings(model='nomic-embed-text')
    sample_vector = embeddings.embed_query("sample text")
    index = faiss.IndexFlatL2(len(sample_vector))
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    all_chunks = []
    for doc in documents:
        chunks = split_document(doc)
        all_chunks.extend(chunks)
    vector_store.add_documents(documents=all_chunks)
    return vector_store

def create_rag_chain(retriever):
    """
    Creates a RAG chain with custom chat model.
    """
    prompt = """
    You are an assistant that selects the best matching AI tool based on user requirements.
    Use the following tool information to answer the query.
    If no tool is relevant, state that no matching tool was found.

    List Top 3 search results within my data only.
    
    Question: {question}
    Tool Information: {context}
    
    Answer in bullet points:
    """
    model = CustomChatOllama(model="deepseek-r1:7b")
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
    Manages the vector store cache.
    """
    global VECTOR_STORE
    if tool_data and len(tool_data) > 0:
        documents = convert_tools_to_documents(tool_data)
        VECTOR_STORE = setup_vector_store(documents)
    elif VECTOR_STORE is None:
        raise ValueError("No vector store available. Provide tool_data at least once.")
    return VECTOR_STORE

def run_rag_system(user_query, tool_data=None):
    """
    Runs the RAG system with the new API.
    """
    vector_store = get_vector_store(tool_data)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    rag_chain = create_rag_chain(retriever)
    
    final_response = ""
    for chunk in rag_chain.stream(user_query):
        final_response += chunk
        print(chunk, end="", flush=True)
    return final_response

def main():
    # Your existing tool_data list here
    tool_data = [
    {
        "name": "VintageImageGen",
        "description": "Generates vintage-style images using AI algorithms.",
        "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
    },
    {
        "name": "ModernImageGen",
        "description": "Generates modern images with high resolution and clarity.",
        "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
    },
    {
        "name": "AIChat",
        "description": "An AI-powered chatbot for conversational purposes.",
        "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"
    },
    {
        "name": "ChatGPT",
        "description": "A conversational AI developed by OpenAI for natural language understanding.",
        "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"
    },
    {
        "name": "DALL-E",
        "description": "Generates creative images from textual descriptions using AI.",
        "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"
    },
    {
        "name": "Midjourney",
        "description": "An AI tool that creates artistic images based on text prompts.",
        "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"
    },
    {
        "name": "StableDiffusion",
        "description": "A latent diffusion model for generating detailed images from text.",
        "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"
    },
    {
        "name": "Copilot",
        "description": "An AI pair programmer that assists with code completion and generation.",
        "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"
    },
    {
        "name": "DeepLTranslate",
        "description": "An AI-powered translation service for multiple languages.",
        "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"
    },
    {
        "name": "VoiceClone",
        "description": "Clones and synthesizes human voices using advanced AI techniques.",
        "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"
    },
    {
        "name": "SentimentAnalyzer",
        "description": "Analyzes text to determine the sentiment using AI.",
        "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"
    },
    {
        "name": "RecommenderAI",
        "description": "Provides personalized recommendations based on user data and AI analysis.",
        "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"
    },
    {
        "name": "FraudDetector",
        "description": "Detects fraudulent activities using sophisticated AI algorithms.",
        "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"
    },
    {
        "name": "AnomalyFinder",
        "description": "Identifies anomalies in datasets using high-sensitivity AI models.",
        "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"
    },
    {
        "name": "VirtualAssistant",
        "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.",
        "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"
    }
]
    
    user_query = input("Enter your tool query: ")
    print("\nRetrieving and generating answer (with tool_data provided)...\n")
    answer = run_rag_system(user_query, tool_data)
    print("\n\nFinal Answer:\n", answer)
    
    user_query2 = input("\nEnter another query (without providing tool_data): ")
    print("\nRetrieving and generating answer (using cached vector store)...\n")
    answer2 = run_rag_system(user_query2)
    print("\n\nFinal Answer:\n", answer2)

if __name__ == "__main__":
    main()
# #!/usr/bin/env python
# import os
# import warnings
# import numpy as np
# import faiss
# from dotenv import load_dotenv

# # Import LangChain components (inspired by previous examples)
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain.docstore.document import Document

# # Environment and warning setup
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# warnings.filterwarnings("ignore")
# load_dotenv()

# # Global variable to cache the vector store
# VECTOR_STORE = None

# def convert_tools_to_documents(tool_data):
#     """
#     Converts a list of dictionaries representing tool data into Document objects.
#     Each dictionary should have keys: 'name', 'description', and 'api_details'.
#     """
#     documents = []
#     for tool in tool_data:
#         content = (
#             f"Name: {tool.get('name', 'N/A')}\n"
#             f"Description: {tool.get('description', 'No description provided.')}\n"
#             f"API Details: {tool.get('api_details', 'N/A')}"
#         )
#         documents.append(Document(page_content=content))
#     return documents

# def format_docs(docs):
#     """
#     Concatenates a list of Document objects into a single string.
#     """
#     return "\n\n".join([doc.page_content for doc in docs])

# def split_document(document, chunk_size=200, overlap=50):
#     """
#     Splits a Document into smaller chunks if its content exceeds chunk_size.
#     Returns a list of Document objects.
#     """
#     content = document.page_content
#     if len(content) <= chunk_size:
#         return [document]
#     chunks = []
#     start = 0
#     while start < len(content):
#         chunk = content[start:start+chunk_size]
#         chunks.append(Document(page_content=chunk))
#         start += chunk_size - overlap
#     return chunks

# def setup_vector_store(documents):
#     """
#     Creates a FAISS vector store from a list of documents (after splitting them into chunks)
#     using an Ollama embedding model.
#     """
#     embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
#     sample_vector = embeddings.embed_query("sample text")
#     index = faiss.IndexFlatL2(len(sample_vector))
    
#     vector_store = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={}
#     )
#     all_chunks = []
#     for doc in documents:
#         chunks = split_document(doc)
#         all_chunks.extend(chunks)
#     vector_store.add_documents(documents=all_chunks)
#     return vector_store

# def create_rag_chain(retriever):
#     """
#     Creates a RAG chain with a prompt template and ChatOllama model.
#     The prompt instructs the model to select the best matching tool based on the context.
#     """
#     prompt = """
#     You are an assistant that selects the best matching AI tool based on user requirements.
#     Use the following tool information to answer the query.
#     If no tool is relevant, state that no matching tool was found.

#     List Top 3 search results within my data only.
    
#     Question: {question}
#     Tool Information: {context}
    
#     Answer in bullet points:
#     """
#     model = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434")
#     prompt_template = ChatPromptTemplate.from_template(prompt)
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )
#     return rag_chain

# def get_vector_store(tool_data=None):
#     """
#     Returns the global VECTOR_STORE. If tool_data is provided (non-empty list),
#     it rebuilds the vector store with the new data. Otherwise, it returns the cached store.
#     """
#     global VECTOR_STORE
#     if tool_data and len(tool_data) > 0:
#         # Build new vector store from the provided tool_data
#         documents = convert_tools_to_documents(tool_data)
#         VECTOR_STORE = setup_vector_store(documents)
#     elif VECTOR_STORE is None:
#         raise ValueError("No vector store available. Provide tool_data at least once.")
#     return VECTOR_STORE

# def run_rag_system(user_query, tool_data=None):
#     """
#     Runs the RAG system: If tool_data is provided, it rebuilds the vector store.
#     Otherwise, it uses the cached vector store to search and answer the query.
#     """
#     vector_store = get_vector_store(tool_data)
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
#     rag_chain = create_rag_chain(retriever)
    
#     final_response = ""
#     for chunk in rag_chain.stream(user_query):
#         final_response += chunk
#         print(chunk, end="", flush=True)
#     return final_response

# def main():
#     # Sample tool data to be used when first building the vector store.
#     tool_data = [
#     {
#         "name": "VintageImageGen",
#         "description": "Generates vintage-style images using AI algorithms.",
#         "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
#     },
#     {
#         "name": "ModernImageGen",
#         "description": "Generates modern images with high resolution and clarity.",
#         "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
#     },
#     {
#         "name": "AIChat",
#         "description": "An AI-powered chatbot for conversational purposes.",
#         "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"
#     },
#     {
#         "name": "ChatGPT",
#         "description": "A conversational AI developed by OpenAI for natural language understanding.",
#         "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"
#     },
#     {
#         "name": "DALL-E",
#         "description": "Generates creative images from textual descriptions using AI.",
#         "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"
#     },
#     {
#         "name": "Midjourney",
#         "description": "An AI tool that creates artistic images based on text prompts.",
#         "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"
#     },
#     {
#         "name": "StableDiffusion",
#         "description": "A latent diffusion model for generating detailed images from text.",
#         "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"
#     },
#     {
#         "name": "Copilot",
#         "description": "An AI pair programmer that assists with code completion and generation.",
#         "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"
#     },
#     {
#         "name": "DeepLTranslate",
#         "description": "An AI-powered translation service for multiple languages.",
#         "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"
#     },
#     {
#         "name": "VoiceClone",
#         "description": "Clones and synthesizes human voices using advanced AI techniques.",
#         "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"
#     },
#     {
#         "name": "SentimentAnalyzer",
#         "description": "Analyzes text to determine the sentiment using AI.",
#         "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"
#     },
#     {
#         "name": "RecommenderAI",
#         "description": "Provides personalized recommendations based on user data and AI analysis.",
#         "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"
#     },
#     {
#         "name": "FraudDetector",
#         "description": "Detects fraudulent activities using sophisticated AI algorithms.",
#         "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"
#     },
#     {
#         "name": "AnomalyFinder",
#         "description": "Identifies anomalies in datasets using high-sensitivity AI models.",
#         "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"
#     },
#     {
#         "name": "VirtualAssistant",
#         "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.",
#         "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"
#     }
# ]

    
#     # Option 1: Rebuild the vector store by providing tool_data
#     # run_rag_system will rebuild if tool_data is provided.
#     user_query = input("Enter your tool query: ")
#     print("\nRetrieving and generating answer (with tool_data provided)...\n")
#     answer = run_rag_system(user_query, tool_data)
#     print("\n\nFinal Answer:\n", answer)
    
#     # Option 2: Directly ask a new question without providing tool_data,
#     # using the already cached vector store.
#     user_query2 = input("\nEnter another query (without providing tool_data): ")
#     print("\nRetrieving and generating answer (using cached vector store)...\n")
#     answer2 = run_rag_system(user_query2)
#     print("\n\nFinal Answer:\n", answer2)

# if __name__ == "__main__":
#     main()
