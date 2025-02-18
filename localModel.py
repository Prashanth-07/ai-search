#!/usr/bin/env python
import os
import warnings
import numpy as np
import faiss
import json
from dotenv import load_dotenv

# Import LangChain components (inspired by previous examples)
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

# Global variable to cache the vector store
VECTOR_STORE = None

def convert_tools_to_documents(tool_data):
    """
    Converts a list of dictionaries representing tool data into Document objects.
    Each dictionary should have keys: 'name', 'description', and 'api_details'.
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
    Creates a RAG chain with an optimized prompt template and ChatOllama model.
    This version includes additional parameters (top_p, min_tokens, temperature,
    presence_penalty, stream, max_tokens, system_prompt) to guide the model for 99%
    accurate responses.
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
        stream=True,
        max_tokens=500,
        system_prompt="You are a highly accurate tool selection assistant who provides concise, fact-based recommendations with expert reasoning."
    )
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return rag_chain


def get_vector_store(tool_data=None, update_mode=False):
    """
    Returns the global VECTOR_STORE. If tool_data is provided (non-empty list),
    it rebuilds the vector store with the new data or updates it if update_mode is True.
    Otherwise, it returns the cached store.
    Includes error handling to prevent program termination.
    """
    global VECTOR_STORE
    try:
        if tool_data and len(tool_data) > 0:
            if VECTOR_STORE is None or not update_mode:
                # Build new vector store from the provided tool_data
                try:
                    documents = convert_tools_to_documents(tool_data)
                    VECTOR_STORE = setup_vector_store(documents)
                except Exception as e:
                    print(f"\nError building vector store: {str(e)}")
                    if VECTOR_STORE is None:
                        raise ValueError(f"Failed to initialize vector store: {str(e)}")
                    # If updating fails but we have an existing store, return it
                    print("Keeping existing vector store.")
            else:
                # Update existing vector store with new documents
                try:
                    documents = convert_tools_to_documents(tool_data)
                    chunks = []
                    for doc in documents:
                        split_chunks = split_document(doc)
                        chunks.extend(split_chunks)
                    VECTOR_STORE.add_documents(documents=chunks)
                except Exception as e:
                    print(f"\nError updating vector store: {str(e)}")
                    print("Vector store update failed, but existing data is preserved.")
        elif VECTOR_STORE is None:
            raise ValueError("No vector store available. Provide tool_data at least once.")
        return VECTOR_STORE
    except Exception as e:
        # Re-raise ValueErrors (like no vector store available)
        if isinstance(e, ValueError):
            raise
        # For other exceptions, provide context but allow recovery
        print(f"\nUnexpected error in vector store management: {str(e)}")
        if VECTOR_STORE is not None:
            print("Returning existing vector store despite error.")
            return VECTOR_STORE
        raise ValueError(f"Vector store unavailable due to error: {str(e)}")

def run_rag_system(user_query, tool_data=None, update_mode=False, silent_mode=False):
    """
    Runs the RAG system: If tool_data is provided, it rebuilds or updates the vector store
    based on update_mode. Otherwise, it uses the cached vector store to search and answer the query.
    Contains comprehensive error handling to prevent program termination.
    
    Parameters:
    - user_query: The query to process
    - tool_data: Optional tool data to update the vector store
    - update_mode: Whether to update (True) or replace (False) existing data
    - silent_mode: If True, suppresses output during vector store updates
    """
    try:
        # Get vector store with error handling
        try:
            vector_store = get_vector_store(tool_data, update_mode)
            # If we're just updating the vector store in silent mode, return early
            if tool_data is not None and silent_mode:
                return "Vector store updated successfully."
        except ValueError as e:
            print(f"\nError accessing vector store: {e}")
            return "Error: Vector store not available. Please initialize with tool data first."
        except Exception as e:
            print(f"\nUnexpected error with vector store: {str(e)}")
            return f"Error while processing vector store: {str(e)}"
            
        # Create retriever with error handling
        try:
            retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
        except Exception as e:
            print(f"\nError creating retriever: {str(e)}")
            return f"Error creating retrieval system: {str(e)}"
            
        # Create RAG chain with error handling
        try:
            rag_chain = create_rag_chain(retriever)
        except Exception as e:
            print(f"\nError creating RAG chain: {str(e)}")
            return f"Error setting up query processing: {str(e)}"
        
        # Process query with error handling
        final_response = ""
        try:
            for chunk in rag_chain.stream(user_query):
                final_response += chunk
                if not silent_mode:
                    print(chunk, end="", flush=True)
            return final_response
        except Exception as e:
            print(f"\nError during query processing: {str(e)}")
            return f"Error while processing your query: {str(e)}"
            
    except Exception as e:
        print(f"\nUnexpected error in RAG system: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

def parse_json_input(json_str):
    """
    Parse a JSON string into a Python object, with error handling.
    Ensures the result is always a list of tool objects.
    """
    try:
        parsed_data = json.loads(json_str)
        
        # If we got a single object instead of a list, wrap it in a list
        if isinstance(parsed_data, dict):
            return [parsed_data]
        elif isinstance(parsed_data, list):
            return parsed_data
        else:
            print(f"Error: JSON must be an object or list of objects, got {type(parsed_data)}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def main():
    # Initial sample tool data
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

    # Initialize the vector store with tool data
    print("Initializing RAG system with tool data...")
    try:
        get_vector_store(tool_data)
        print("Vector store initialized successfully!")
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        print("Trying to continue with limited functionality.")
    
    # Start a continuous loop for queries
    print("\n===== RAG QUERY SYSTEM =====")
    print("Special commands:")
    print("  - 'exit' or 'quit': Terminate the program")
    print("  - 'add tool': Add new tool data")
    print("  - 'reset tools': Reset tool data and rebuild vector store")
    print("  - 'help': Show these commands")
    
    while True:
        try:
            user_input = input("\nEnter your tool query or command: ")
            
            # Check for commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting RAG system. Goodbye!")
                break
                
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - 'exit' or 'quit': Terminate the program")
                print("  - 'add tool': Add new tool data")
                print("  - 'reset tools': Reset tool data and rebuild vector store")
                print("  - 'help': Show these commands")
                continue
                
            elif user_input.lower() == 'add tool':
                try:
                    print("\nEnter new tool data in JSON format.")
                    print("Example: [{\"name\": \"NewTool\", \"description\": \"Tool description\", \"api_details\": \"API info\"}]")
                    print("Enter 'cancel' to abort adding tools.")
                    
                    json_input = input("\nNew tool data (JSON): ")
                    if json_input.lower() == 'cancel':
                        print("Tool addition cancelled.")
                        continue
                        
                    new_tools = parse_json_input(json_input)
                    if new_tools:
                        # Validate tool structure before adding
                        valid_tools = []
                        for tool in new_tools:
                            if not isinstance(tool, dict):
                                print(f"Error: Skipping invalid tool (not an object): {tool}")
                                continue
                                
                            # Ensure the tool has at least a name
                            if 'name' not in tool:
                                print(f"Error: Skipping tool without a name: {tool}")
                                continue
                                
                            # Add default values for missing fields
                            if 'description' not in tool:
                                tool['description'] = "No description provided."
                            if 'api_details' not in tool:
                                tool['api_details'] = "No API details provided."
                                
                            valid_tools.append(tool)
                        
                        if valid_tools:
                            print("\nAdding new tools to the vector store...")
                            try:
                                # Use silent_mode=True to prevent RAG from running query
                                run_rag_system("dummy query - adding tools", valid_tools, update_mode=True, silent_mode=True)
                                print("\nTools added successfully!")
                            except Exception as e:
                                print(f"\nError adding tools: {str(e)}")
                                print("Tools could not be added. Please try again.")
                        else:
                            print("\nNo valid tools to add.")
                except Exception as e:
                    print(f"\nUnexpected error during tool addition: {str(e)}")
                    print("Tool addition failed. Please try again.")
                continue
                
            elif user_input.lower() == 'reset tools':
                try:
                    print("\nEnter new tool data in JSON format to reset the vector store,")
                    print("or press Enter to reset with the default tool set.")
                    print("Enter 'cancel' to abort reset.")
                    
                    json_input = input("\nNew tool data for reset (JSON or press Enter): ")
                    if json_input.lower() == 'cancel':
                        print("Reset cancelled.")
                        continue
                        
                    if json_input.strip():
                        reset_tools = parse_json_input(json_input)
                        if not reset_tools:
                            continue
                    else:
                        reset_tools = tool_data
                        
                    print("\nResetting vector store with new tool data...")
                    try:
                        # Use silent_mode=True to prevent RAG from running query
                        run_rag_system("dummy query - resetting tools", reset_tools, update_mode=False, silent_mode=True)
                        print("\nVector store reset successfully!")
                    except Exception as e:
                        print(f"\nError resetting vector store: {str(e)}")
                        print("Reset failed. Previous data may still be available.")
                except Exception as e:
                    print(f"\nUnexpected error during reset operation: {str(e)}")
                    print("Reset operation failed. Please try again.")
                continue
                
            # Process normal user query
            try:
                print("\nRetrieving and generating answer...\n")
                answer = run_rag_system(user_input)
                print("\n\nFinal Answer:\n", answer)
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                print("Unable to complete your query. Please try again or use a different query.")
                
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Exiting...")
            break
        except EOFError:
            print("\n\nEOF detected. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("The system encountered an error but will continue running.")
            print("Please try again with a different query or command.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nCritical error in main program: {str(e)}")
        print("The program encountered a critical error and must exit.")
        print("Please check your environment setup, especially Ollama availability.")
        print("\nTroubleshooting tips:")
        print("1. Ensure Ollama is running at http://localhost:11434")
        print("2. Check that required models are available (nomic-embed-text, deepseek-r1:1.5b)")
        print("3. Verify network connectivity if using remote Ollama instance")
        print("4. Check system resources (memory, disk space)")
        print("\nPress Enter to exit...")
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
#     Creates a RAG chain with an optimized prompt template and ChatOllama model.
#     This version includes additional parameters (top_p, min_tokens, temperature,
#     presence_penalty, stream, max_tokens, system_prompt) to guide the model for 99%
#     accurate responses.
#     """
#     prompt = """
# You are an expert AI assistant specializing in selecting the best matching tool from a given list based on user requirements. Analyze the user query carefully and match it with the provided tool information. Consider the tool name, description, and API details to determine the top three most relevant tools. Provide your answer in clear bullet points with detailed reasoning. If no tool is highly relevant, explicitly state that no matching tool was found.

# Your answer must follow this format:
# - **Tool Name**: [Name]
#   - *Description*: [Brief description and why it matches]
#   - *API Details*: [Relevant API info]

# Question: {question}

# Tool Information: {context}
#     """
#     model = ChatOllama(
#         model="deepseek-r1:1.5b",
#         base_url="http://localhost:11434",
#         top_p=0.9,
#         min_tokens=50,
#         temperature=0.3,
#         presence_penalty=0.0,
#         stream=True,
#         max_tokens=500,
#         system_prompt="You are a highly accurate tool selection assistant who provides concise, fact-based recommendations with expert reasoning."
#     )
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

#     # Initialize the vector store with tool data
#     print("Initializing RAG system with tool data...")
#     get_vector_store(tool_data)
#     print("Vector store initialized successfully!")
    
#     # Start a continuous loop for queries
#     print("\n===== RAG QUERY SYSTEM =====")
#     print("Enter 'exit' or 'quit' to terminate the program")
    
#     while True:
#         user_query = input("\nEnter your tool query: ")
        
#         # Check for exit commands
#         if user_query.lower() in ['exit', 'quit', 'q']:
#             print("Exiting RAG system. Goodbye!")
#             break
            
#         print("\nRetrieving and generating answer...\n")
#         answer = run_rag_system(user_query)
#         print("\n\nFinal Answer:\n", answer)

# if __name__ == "__main__":
#     main()
# # #!/usr/bin/env python
# # import os
# # import warnings
# # import numpy as np
# # import faiss
# # from dotenv import load_dotenv

# # # Import LangChain components (inspired by previous examples)
# # from langchain_ollama import OllamaEmbeddings, ChatOllama
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.docstore.in_memory import InMemoryDocstore
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.runnables import RunnablePassthrough
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain.docstore.document import Document

# # # Environment and warning setup
# # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # warnings.filterwarnings("ignore")
# # load_dotenv()

# # # Global variable to cache the vector store
# # VECTOR_STORE = None

# # def convert_tools_to_documents(tool_data):
# #     """
# #     Converts a list of dictionaries representing tool data into Document objects.
# #     Each dictionary should have keys: 'name', 'description', and 'api_details'.
# #     """
# #     documents = []
# #     for tool in tool_data:
# #         content = (
# #             f"Name: {tool.get('name', 'N/A')}\n"
# #             f"Description: {tool.get('description', 'No description provided.')}\n"
# #             f"API Details: {tool.get('api_details', 'N/A')}"
# #         )
# #         documents.append(Document(page_content=content))
# #     return documents

# # def format_docs(docs):
# #     """
# #     Concatenates a list of Document objects into a single string.
# #     """
# #     return "\n\n".join([doc.page_content for doc in docs])

# # def split_document(document, chunk_size=200, overlap=50):
# #     """
# #     Splits a Document into smaller chunks if its content exceeds chunk_size.
# #     Returns a list of Document objects.
# #     """
# #     content = document.page_content
# #     if len(content) <= chunk_size:
# #         return [document]
# #     chunks = []
# #     start = 0
# #     while start < len(content):
# #         chunk = content[start:start+chunk_size]
# #         chunks.append(Document(page_content=chunk))
# #         start += chunk_size - overlap
# #     return chunks

# # def setup_vector_store(documents):
# #     """
# #     Creates a FAISS vector store from a list of documents (after splitting them into chunks)
# #     using an Ollama embedding model.
# #     """
# #     embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
# #     sample_vector = embeddings.embed_query("sample text")
# #     index = faiss.IndexFlatL2(len(sample_vector))
    
# #     vector_store = FAISS(
# #         embedding_function=embeddings,
# #         index=index,
# #         docstore=InMemoryDocstore(),
# #         index_to_docstore_id={}
# #     )
# #     all_chunks = []
# #     for doc in documents:
# #         chunks = split_document(doc)
# #         all_chunks.extend(chunks)
# #     vector_store.add_documents(documents=all_chunks)
# #     return vector_store

# # def create_rag_chain(retriever):
# #     """
# #     Creates a RAG chain with an optimized prompt template and ChatOllama model.
# #     This version includes additional parameters (top_p, min_tokens, temperature,
# #     presence_penalty, stream, max_tokens, system_prompt) to guide the model for 99%
# #     accurate responses.
# #     """
# #     prompt = """
# # You are an expert AI assistant specializing in selecting the best matching tool from a given list based on user requirements. Analyze the user query carefully and match it with the provided tool information. Consider the tool name, description, and API details to determine the top three most relevant tools. Provide your answer in clear bullet points with detailed reasoning. If no tool is highly relevant, explicitly state that no matching tool was found.

# # Your answer must follow this format:
# # - **Tool Name**: [Name]
# #   - *Description*: [Brief description and why it matches]
# #   - *API Details*: [Relevant API info]

# # Question: {question}

# # Tool Information: {context}
# #     """
# #     model = ChatOllama(
# #         model="deepseek-r1:1.5b",
# #         base_url="http://localhost:11434",
# #         top_p=0.9,
# #         min_tokens=50,
# #         temperature=0.3,
# #         presence_penalty=0.0,
# #         stream=True,
# #         max_tokens=500,
# #         system_prompt="You are a highly accurate tool selection assistant who provides concise, fact-based recommendations with expert reasoning."
# #     )
# #     prompt_template = ChatPromptTemplate.from_template(prompt)
    
# #     rag_chain = (
# #         {"context": retriever | format_docs, "question": RunnablePassthrough()}
# #         | prompt_template
# #         | model
# #         | StrOutputParser()
# #     )
# #     return rag_chain


# # def get_vector_store(tool_data=None):
# #     """
# #     Returns the global VECTOR_STORE. If tool_data is provided (non-empty list),
# #     it rebuilds the vector store with the new data. Otherwise, it returns the cached store.
# #     """
# #     global VECTOR_STORE
# #     if tool_data and len(tool_data) > 0:
# #         # Build new vector store from the provided tool_data
# #         documents = convert_tools_to_documents(tool_data)
# #         VECTOR_STORE = setup_vector_store(documents)
# #     elif VECTOR_STORE is None:
# #         raise ValueError("No vector store available. Provide tool_data at least once.")
# #     return VECTOR_STORE

# # def run_rag_system(user_query, tool_data=None):
# #     """
# #     Runs the RAG system: If tool_data is provided, it rebuilds the vector store.
# #     Otherwise, it uses the cached vector store to search and answer the query.
# #     """
# #     vector_store = get_vector_store(tool_data)
# #     retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
# #     rag_chain = create_rag_chain(retriever)
    
# #     final_response = ""
# #     for chunk in rag_chain.stream(user_query):
# #         final_response += chunk
# #         print(chunk, end="", flush=True)
# #     return final_response

# # def main():
# #     # Sample tool data to be used when first building the vector store.
# #     tool_data = [
# #     {
# #         "name": "VintageImageGen",
# #         "description": "Generates vintage-style images using AI algorithms.",
# #         "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
# #     },
# #     {
# #         "name": "ModernImageGen",
# #         "description": "Generates modern images with high resolution and clarity.",
# #         "api_details": "Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
# #     },
# #     {
# #         "name": "AIChat",
# #         "description": "An AI-powered chatbot for conversational purposes.",
# #         "api_details": "Endpoint: /api/chat, Method: POST, Params: {'language': 'en'}"
# #     },
# #     {
# #         "name": "ChatGPT",
# #         "description": "A conversational AI developed by OpenAI for natural language understanding.",
# #         "api_details": "Endpoint: /api/chatgpt, Method: POST, Params: {'version': 'latest'}"
# #     },
# #     {
# #         "name": "DALL-E",
# #         "description": "Generates creative images from textual descriptions using AI.",
# #         "api_details": "Endpoint: /api/dalle, Method: POST, Params: {'version': '2'}"
# #     },
# #     {
# #         "name": "Midjourney",
# #         "description": "An AI tool that creates artistic images based on text prompts.",
# #         "api_details": "Endpoint: /api/midjourney, Method: POST, Params: {'quality': 'high'}"
# #     },
# #     {
# #         "name": "StableDiffusion",
# #         "description": "A latent diffusion model for generating detailed images from text.",
# #         "api_details": "Endpoint: /api/stable, Method: POST, Params: {'steps': 50}"
# #     },
# #     {
# #         "name": "Copilot",
# #         "description": "An AI pair programmer that assists with code completion and generation.",
# #         "api_details": "Endpoint: /api/copilot, Method: POST, Params: {'language': 'python'}"
# #     },
# #     {
# #         "name": "DeepLTranslate",
# #         "description": "An AI-powered translation service for multiple languages.",
# #         "api_details": "Endpoint: /api/deepl, Method: POST, Params: {'target_language': 'en'}"
# #     },
# #     {
# #         "name": "VoiceClone",
# #         "description": "Clones and synthesizes human voices using advanced AI techniques.",
# #         "api_details": "Endpoint: /api/voice, Method: POST, Params: {'gender': 'neutral'}"
# #     },
# #     {
# #         "name": "SentimentAnalyzer",
# #         "description": "Analyzes text to determine the sentiment using AI.",
# #         "api_details": "Endpoint: /api/sentiment, Method: POST, Params: {'language': 'en'}"
# #     },
# #     {
# #         "name": "RecommenderAI",
# #         "description": "Provides personalized recommendations based on user data and AI analysis.",
# #         "api_details": "Endpoint: /api/recommender, Method: POST, Params: {'user_id': 'string'}"
# #     },
# #     {
# #         "name": "FraudDetector",
# #         "description": "Detects fraudulent activities using sophisticated AI algorithms.",
# #         "api_details": "Endpoint: /api/fraud, Method: POST, Params: {'threshold': 0.8}"
# #     },
# #     {
# #         "name": "AnomalyFinder",
# #         "description": "Identifies anomalies in datasets using high-sensitivity AI models.",
# #         "api_details": "Endpoint: /api/anomaly, Method: POST, Params: {'sensitivity': 'high'}"
# #     },
# #     {
# #         "name": "VirtualAssistant",
# #         "description": "A comprehensive virtual assistant powered by AI to manage tasks and provide information.",
# #         "api_details": "Endpoint: /api/assistant, Method: POST, Params: {'capabilities': 'full'}"
# #     }
# # ]

    
# #     # Option 1: Rebuild the vector store by providing tool_data
# #     # run_rag_system will rebuild if tool_data is provided.
# #     user_query = input("Enter your tool query: ")
# #     print("\nRetrieving and generating answer (with tool_data provided)...\n")
# #     answer = run_rag_system(user_query, tool_data)
# #     print("\n\nFinal Answer:\n", answer)
    
# #     # Option 2: Directly ask a new question without providing tool_data,
# #     # using the already cached vector store.
# #     user_query2 = input("\nEnter another query (without providing tool_data): ")
# #     print("\nRetrieving and generating answer (using cached vector store)...\n")
# #     answer2 = run_rag_system(user_query2)
# #     print("\n\nFinal Answer:\n", answer2)

# # if __name__ == "__main__":
# #     main()
