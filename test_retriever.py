#!/usr/bin/env python
import os
import warnings
import json
import faiss
from dotenv import load_dotenv
from langchain.docstore.document import Document
# Import LangChain components
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# Environment and warning setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Global variable to cache the vector store
VECTOR_STORE = None

def convert_tools_to_documents(tool_data):
    """
    Converts a list of dictionaries (tool data) into Document objects.
    Each JSON is treated as an atomic unit and will not be split further.
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

def group_documents_by_json_boundary(documents, max_chars=300):
    """
    Groups a list of Document objects (each representing one JSON tool)
    into larger Document objects without breaking any individual JSON.
    The max_chars threshold determines how many characters can be grouped
    together before starting a new group. A lower threshold (e.g., 300)
    preserves more granular boundaries.
    """
    grouped_docs = []
    current_group = ""
    for doc in documents:
        # If adding this document would exceed the threshold, start a new group.
        if current_group and (len(current_group) + len(doc.page_content) + 2 > max_chars):
            grouped_docs.append(Document(page_content=current_group))
            current_group = doc.page_content
        else:
            if current_group:
                current_group += "\n\n" + doc.page_content
            else:
                current_group = doc.page_content
    if current_group:
        grouped_docs.append(Document(page_content=current_group))
    return grouped_docs

def setup_vector_store(tool_data):
    """
    Creates a FAISS vector store from a list of tool JSON objects.
    Instead of arbitrary splitting by characters, we convert each tool's JSON
    into a Document and group them only at natural JSON boundaries using a low threshold.
    """
    docs = convert_tools_to_documents(tool_data)
    # Group documents without merging too many; using 300 characters preserves individual entries.
    grouped_docs = group_documents_by_json_boundary(docs, max_chars=300)
    
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    sample_vector = embeddings.embed_query("sample text")
    index = faiss.IndexFlatL2(len(sample_vector))
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=grouped_docs)
    return vector_store

def get_vector_store(tool_data=None):
    """
    Returns the global VECTOR_STORE. If tool_data is provided,
    it rebuilds the vector store with the new data.
    """
    global VECTOR_STORE
    if tool_data and len(tool_data) > 0:
        VECTOR_STORE = setup_vector_store(tool_data)
    elif VECTOR_STORE is None:
        raise ValueError("No vector store available. Provide tool_data at least once.")
    return VECTOR_STORE

def test_retriever(user_query, tool_data):
    """
    Builds the vector store and prints the documents retrieved for the given query.
    Uses retriever settings that aim to return exactly the top k (here, k=3) documents.
    """
    vector_store = get_vector_store(tool_data)
    # Set k=3, and for MMR, also set fetch_k to k so that we don't over-retrieve.
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': 3, 'fetch_k': 3, 'lambda_mult': 1.0}
    )
    docs = retriever.get_relevant_documents(user_query)
    print("\n--- Retrieved Documents ---")
    for idx, doc in enumerate(docs):
        print(f"Document {idx+1}:")
        print(doc.page_content)
        print("-" * 40)
    return docs

if __name__ == "__main__":
    # Sample tool data (each tool is a JSON object)
    tool_data = [
        {"name": "VintageImageGen", "description": "Generates vintage-style images using AI algorithms.", "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"},
        {"name": "ModernImageGen", "description": "Generates modern images with high resolution and clarity.", "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"},
        {"name": "AIChat", "description": "An AI-powered chatbot for conversational purposes.", "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"},
        {"name": "ChatGPT", "description": "A conversational AI developed by OpenAI for natural language understanding.", "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"},
        {"name": "DALL-E", "description": "Generates creative images from textual descriptions using AI.", "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"},
        {"name": "Midjourney", "description": "An AI tool that creates artistic images based on text prompts.", "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"},
        {"name": "StableDiffusion", "description": "A latent diffusion model for generating detailed images from text.", "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"},
        {"name": "Copilot", "description": "An AI pair programmer that assists with code completion and generation.", "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"},
        {"name": "DeepLTranslate", "description": "An AI-powered translation service for multiple languages.", "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"},
        {"name": "VoiceClone", "description": "Clones and synthesizes human voices using advanced AI techniques.", "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"},
        {"name": "SentimentAnalyzer", "description": "Analyzes text to determine the sentiment using AI.", "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"},
        {"name": "RecommenderAI", "description": "Provides personalized recommendations based on user data and AI analysis.", "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"},
        {"name": "FraudDetector", "description": "Detects fraudulent activities using sophisticated AI algorithms.", "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"},
        {"name": "AnomalyFinder", "description": "Identifies anomalies in datasets using high-sensitivity AI models.", "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"},
        {"name": "VirtualAssistant", "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.", "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"}
    ]
    
    print("===== RETRIEVER DEBUG TEST =====")
    query = input("Enter your tool query: ")
    test_retriever(query, tool_data)

# #!/usr/bin/env python
# import os
# import warnings
# import json
# import faiss
# from dotenv import load_dotenv
# from langchain.docstore.document import Document
# # Import LangChain components
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore

# # Environment and warning setup
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# warnings.filterwarnings("ignore")
# load_dotenv()

# # Global variable to cache the vector store
# VECTOR_STORE = None

# def convert_tools_to_documents(tool_data):
#     """
#     Converts a list of dictionaries representing tool data into Document objects.
#     Each dictionary (JSON) is treated as an atomic unit and will not be split further.
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

# def group_documents_by_json_boundary(documents, max_chars=1000):
#     """
#     Groups a list of Document objects (each representing one JSON tool)
#     into larger Document objects without breaking any individual JSON.
#     The max_chars threshold determines how many characters (approximate)
#     can be grouped together before starting a new group.
#     """
#     grouped_docs = []
#     current_group = ""
#     for doc in documents:
#         # If adding this document would exceed the threshold, start a new group.
#         if current_group and (len(current_group) + len(doc.page_content) + 2 > max_chars):
#             grouped_docs.append(Document(page_content=current_group))
#             current_group = doc.page_content
#         else:
#             if current_group:
#                 current_group += "\n\n" + doc.page_content
#             else:
#                 current_group = doc.page_content
#     if current_group:
#         grouped_docs.append(Document(page_content=current_group))
#     return grouped_docs

# def setup_vector_store(tool_data):
#     """
#     Creates a FAISS vector store from a list of tool JSON objects.
#     Instead of splitting arbitrarily by characters, we convert each tool's JSON
#     into a Document and group them together by natural boundaries (i.e. complete JSON entries).
#     """
#     docs = convert_tools_to_documents(tool_data)
#     # Group the documents using a max character threshold.
#     grouped_docs = group_documents_by_json_boundary(docs, max_chars=1000)
    
#     embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
#     sample_vector = embeddings.embed_query("sample text")
#     index = faiss.IndexFlatL2(len(sample_vector))
    
#     vector_store = FAISS(
#         embedding_function=embeddings,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={}
#     )
#     vector_store.add_documents(documents=grouped_docs)
#     return vector_store

# def get_vector_store(tool_data=None):
#     """
#     Returns the global VECTOR_STORE. If tool_data is provided,
#     it rebuilds the vector store with the new data.
#     """
#     global VECTOR_STORE
#     if tool_data and len(tool_data) > 0:
#         documents = tool_data  # tool_data is already a list of JSON objects
#         VECTOR_STORE = setup_vector_store(documents)
#     elif VECTOR_STORE is None:
#         raise ValueError("No vector store available. Provide tool_data at least once.")
#     return VECTOR_STORE

# def test_retriever(user_query, tool_data):
#     """
#     Builds the vector store and prints the documents retrieved for the given query.
#     """
#     vector_store = get_vector_store(tool_data)
#     # Create retriever with chosen parameters (using "mmr" for diversity)
#     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 1})
#     docs = retriever.get_relevant_documents(user_query)
#     print("\n--- Retrieved Documents ---")
#     for idx, doc in enumerate(docs):
#         print(f"Document {idx+1}:")
#         print(doc.page_content)
#         print("-" * 40)
#     return docs

# if __name__ == "__main__":
#     # Sample tool data (each tool is a JSON object)
#     tool_data = [
#         {"name": "VintageImageGen", "description": "Generates vintage-style images using AI algorithms.", "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"},
#         {"name": "ModernImageGen", "description": "Generates modern images with high resolution and clarity.", "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"},
#         {"name": "AIChat", "description": "An AI-powered chatbot for conversational purposes.", "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"},
#         {"name": "ChatGPT", "description": "A conversational AI developed by OpenAI for natural language understanding.", "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"},
#         {"name": "DALL-E", "description": "Generates creative images from textual descriptions using AI.", "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"},
#         {"name": "Midjourney", "description": "An AI tool that creates artistic images based on text prompts.", "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"},
#         {"name": "StableDiffusion", "description": "A latent diffusion model for generating detailed images from text.", "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"},
#         {"name": "Copilot", "description": "An AI pair programmer that assists with code completion and generation.", "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"},
#         {"name": "DeepLTranslate", "description": "An AI-powered translation service for multiple languages.", "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"},
#         {"name": "VoiceClone", "description": "Clones and synthesizes human voices using advanced AI techniques.", "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"},
#         {"name": "SentimentAnalyzer", "description": "Analyzes text to determine the sentiment using AI.", "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"},
#         {"name": "RecommenderAI", "description": "Provides personalized recommendations based on user data and AI analysis.", "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"},
#         {"name": "FraudDetector", "description": "Detects fraudulent activities using sophisticated AI algorithms.", "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"},
#         {"name": "AnomalyFinder", "description": "Identifies anomalies in datasets using high-sensitivity AI models.", "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"},
#         {"name": "VirtualAssistant", "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.", "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"}
#     ]
    
#     print("===== RETRIEVER DEBUG TEST =====")
#     query = input("Enter your tool query: ")
#     test_retriever(query, tool_data)
