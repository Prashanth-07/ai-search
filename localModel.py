#!/usr/bin/env python
import os
import warnings
import numpy as np
import faiss
import json
from dotenv import load_dotenv

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

def setup_vector_store(documents):
    """
    Creates a FAISS vector store from a list of documents 
    using an Ollama embedding model and document grouping with JSON boundaries.
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
    
    # Group documents by natural JSON boundaries
    grouped_docs = group_documents_by_json_boundary(documents, max_chars=300)
    vector_store.add_documents(documents=grouped_docs)
    return vector_store

def create_rag_chain(retriever):
    """
    Creates a RAG chain with an optimized prompt template and ChatOllama model.
    This version includes additional parameters (top_p, min_tokens, temperature,
    presence_penalty, stream, max_tokens, system_prompt) to guide the model for 99%
    accurate responses.
    """
    prompt = """You are an AI assistant specialized in recommending AI tools based on user needs. You have access to a predefined list of AI tools, each with its own capabilities and API details. When a user provides a query about their use case, your task is to identify and suggest the most relevant AI tools from the given dataset.

User Query: {question}

Tool Data: {context}

Your Responsibilities:
1. Analyze the user query to understand their intent and specific requirements.
2. Match the query with relevant tools from the given dataset (Tool Data).
3. Provide the best recommendations, listing the most suitable tools along with a brief description.
4. Explain your choice concisely, ensuring clarity and usefulness.
5. If multiple tools are relevant, rank them in order of suitability.
6. If no tool is a perfect match, suggest the closest alternatives and explain why they might still be useful.
7. If needed, ask a follow-up question to clarify the user’s intent.

Response Format:
- If a single tool is a perfect match:
  - Tool Name: [Brief Description] – Why it's the best match.
- If multiple tools match:
  - Tool 1 Name: [Brief Description] – Why it fits.
  - Tool 2 Name: [Brief Description] – Alternative option.
- If no exact match:
  - "There is no exact match, but here are some alternatives:"
  - Alternative 1 Name: [Description] – How it partially meets the need.

Stay within the given Tool Data, ensure relevance, and provide clear, user-friendly recommendations."""
    model = ChatOllama(
model="deepseek-r1:1.5b",
base_url="http://localhost:11434",
top_p=0.9,
temperature=0.7,
presence_penalty=0.2,
frequency_penalty=0.3,
stream=True,
max_tokens=150,
system_prompt = """You are an intelligent AI assistant specializing in recommending AI tools based on user queries. Your goal is to:
- Understand the user query ({question})
- Analyze the available tool dataset ({context})
- Find the best tool(s) that match the user's needs
- Explain the recommendation concisely and clearly
- Ask clarifying questions if needed

Guidelines:
1. Extract keywords and intent from {question}.
2. Compare the request against {context} (tool dataset) and rank tools by relevance.
3. Provide a justified recommendation for each tool.
4. If there is no exact match, suggest the closest alternatives.
5. Do not generate information outside {context}.
6. Avoid overly technical jargon unless the user requests API details.

Response Format:
- If a single tool is a perfect match:
  - Tool Name: [Brief Description] – Why it's the best match.
- If multiple tools match:
  - Tool 1 Name: [Brief Description] – Why it fits.
  - Tool 2 Name: [Brief Description] – Alternative option.
- If no exact match:
  - "There is no exact match, but here are some alternatives:"
  - Alternative 1 Name: [Description] – How it partially meets the need.

Example Responses:
1. User Query: "I need an AI that generates high-resolution images."
   Response:
     ModernImageGen: Generates modern images with high resolution. Best choice for your request.
2. User Query: "I want a chatbot for my e-commerce website."
   Response:
     AIChat: An AI-powered chatbot for general conversations.
     ChatGPT: A more advanced AI with deep natural language understanding.
3. User Query: "Which API should I use for vintage-style image generation?"
   Response:
     VintageImageGen: For vintage-style images. API Details: Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}."""

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
                    # Group documents by JSON boundaries for better semantics
                    grouped_docs = group_documents_by_json_boundary(documents, max_chars=300)
                    VECTOR_STORE.add_documents(documents=grouped_docs)
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
            
        # Create retriever with error handling - using MMR for better diversity
        try:
            retriever = vector_store.as_retriever(
                search_type="mmr", 
                search_kwargs={'k': 3, 'fetch_k': 3, 'lambda_mult': 1.0}
            )
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
        "description": "Generates vintage-style images.",
        "api_details": "Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
    },
    {
        "name": "ModernImageGen",
        "description": "Generates modern images with high resolution",
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
# import json
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
#     prompt = """User Query: {question}

# Tool Data: {context}

# Instructions:
# Analyze the user's query using advanced semantic understanding to determine its true intent—even if the query is imperfectly written. Compare the semantic meaning of the query with the provided Tool Data, focusing on each tool's primary function. Identify the tool(s) whose core purpose best address the user's needs, and output a bullet-point list as follows:

# - **Tool Name**
# - *Description*: A brief explanation of the tool’s purpose.
# - *API Details*: A concise summary of the API endpoint and key parameters.
# - *Detailed Reasoning*: A brief explanation of why this tool best matches the user's intent.

# If none of the tools' primary functions match the query, output exactly: "No matching tool found."""
#     model = ChatOllama(
# model="deepseek-r1:1.5b",
# base_url="http://localhost:11434",
# top_p=0.9,
# min_tokens=50,
# temperature=0.3,
# presence_penalty=0.0,
# stream=True,
# max_tokens=500,
# system_prompt="""You are an expert AI tool selection assistant who uses advanced semantic analysis to understand the user's intent—even if the query contains spelling errors, broken grammar, or informal language. Your task is to compare the semantic meaning of the user's query with the provided Tool Information (each containing a tool's name, description, and API details) and identify the tool(s) whose primary function best matches the user's needs.

# For each tool, evaluate the semantic similarity between the query's intent and the tool’s core purpose. Give priority to tools whose fundamental functionality directly aligns with what the user is asking for.

# Your final answer must be presented in clear bullet points. For each selected tool, include:
# - **Tool Name**
# - *Description*: A brief explanation of the tool’s primary function.
# - *API Details*: A summary of the API endpoint and key parameters.
# - *Detailed Reasoning*: A concise explanation of why this tool best matches the user's intent.

# If none of the tools’ primary functions match the user's intent, output exactly: "No matching tool found."

# Do not include any internal chain-of-thought or extraneous reasoning details in your final answer."""
#     )
#     prompt_template = ChatPromptTemplate.from_template(prompt)
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )
#     return rag_chain


# def get_vector_store(tool_data=None, update_mode=False):
#     """
#     Returns the global VECTOR_STORE. If tool_data is provided (non-empty list),
#     it rebuilds the vector store with the new data or updates it if update_mode is True.
#     Otherwise, it returns the cached store.
#     Includes error handling to prevent program termination.
#     """
#     global VECTOR_STORE
#     try:
#         if tool_data and len(tool_data) > 0:
#             if VECTOR_STORE is None or not update_mode:
#                 # Build new vector store from the provided tool_data
#                 try:
#                     documents = convert_tools_to_documents(tool_data)
#                     VECTOR_STORE = setup_vector_store(documents)
#                 except Exception as e:
#                     print(f"\nError building vector store: {str(e)}")
#                     if VECTOR_STORE is None:
#                         raise ValueError(f"Failed to initialize vector store: {str(e)}")
#                     # If updating fails but we have an existing store, return it
#                     print("Keeping existing vector store.")
#             else:
#                 # Update existing vector store with new documents
#                 try:
#                     documents = convert_tools_to_documents(tool_data)
#                     chunks = []
#                     for doc in documents:
#                         split_chunks = split_document(doc)
#                         chunks.extend(split_chunks)
#                     VECTOR_STORE.add_documents(documents=chunks)
#                 except Exception as e:
#                     print(f"\nError updating vector store: {str(e)}")
#                     print("Vector store update failed, but existing data is preserved.")
#         elif VECTOR_STORE is None:
#             raise ValueError("No vector store available. Provide tool_data at least once.")
#         return VECTOR_STORE
#     except Exception as e:
#         # Re-raise ValueErrors (like no vector store available)
#         if isinstance(e, ValueError):
#             raise
#         # For other exceptions, provide context but allow recovery
#         print(f"\nUnexpected error in vector store management: {str(e)}")
#         if VECTOR_STORE is not None:
#             print("Returning existing vector store despite error.")
#             return VECTOR_STORE
#         raise ValueError(f"Vector store unavailable due to error: {str(e)}")

# def run_rag_system(user_query, tool_data=None, update_mode=False, silent_mode=False):
#     """
#     Runs the RAG system: If tool_data is provided, it rebuilds or updates the vector store
#     based on update_mode. Otherwise, it uses the cached vector store to search and answer the query.
#     Contains comprehensive error handling to prevent program termination.
    
#     Parameters:
#     - user_query: The query to process
#     - tool_data: Optional tool data to update the vector store
#     - update_mode: Whether to update (True) or replace (False) existing data
#     - silent_mode: If True, suppresses output during vector store updates
#     """
#     try:
#         # Get vector store with error handling
#         try:
#             vector_store = get_vector_store(tool_data, update_mode)
#             # If we're just updating the vector store in silent mode, return early
#             if tool_data is not None and silent_mode:
#                 return "Vector store updated successfully."
#         except ValueError as e:
#             print(f"\nError accessing vector store: {e}")
#             return "Error: Vector store not available. Please initialize with tool data first."
#         except Exception as e:
#             print(f"\nUnexpected error with vector store: {str(e)}")
#             return f"Error while processing vector store: {str(e)}"
            
#         # Create retriever with error handling
#         try:
#             retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
#         except Exception as e:
#             print(f"\nError creating retriever: {str(e)}")
#             return f"Error creating retrieval system: {str(e)}"
            
#         # Create RAG chain with error handling
#         try:
#             rag_chain = create_rag_chain(retriever)
#         except Exception as e:
#             print(f"\nError creating RAG chain: {str(e)}")
#             return f"Error setting up query processing: {str(e)}"
        
#         # Process query with error handling
#         final_response = ""
#         try:
#             for chunk in rag_chain.stream(user_query):
#                 final_response += chunk
#                 if not silent_mode:
#                     print(chunk, end="", flush=True)
#             return final_response
#         except Exception as e:
#             print(f"\nError during query processing: {str(e)}")
#             return f"Error while processing your query: {str(e)}"
            
#     except Exception as e:
#         print(f"\nUnexpected error in RAG system: {str(e)}")
#         return f"An unexpected error occurred: {str(e)}"

# def parse_json_input(json_str):
#     """
#     Parse a JSON string into a Python object, with error handling.
#     Ensures the result is always a list of tool objects.
#     """
#     try:
#         parsed_data = json.loads(json_str)
        
#         # If we got a single object instead of a list, wrap it in a list
#         if isinstance(parsed_data, dict):
#             return [parsed_data]
#         elif isinstance(parsed_data, list):
#             return parsed_data
#         else:
#             print(f"Error: JSON must be an object or list of objects, got {type(parsed_data)}")
#             return None
#     except json.JSONDecodeError as e:
#         print(f"Error parsing JSON: {e}")
#         return None

# def main():
#     # Initial sample tool data
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
#     try:
#         get_vector_store(tool_data)
#         print("Vector store initialized successfully!")
#     except Exception as e:
#         print(f"Error initializing vector store: {str(e)}")
#         print("Trying to continue with limited functionality.")
    
#     # Start a continuous loop for queries
#     print("\n===== RAG QUERY SYSTEM =====")
#     print("Special commands:")
#     print("  - 'exit' or 'quit': Terminate the program")
#     print("  - 'add tool': Add new tool data")
#     print("  - 'reset tools': Reset tool data and rebuild vector store")
#     print("  - 'help': Show these commands")
    
#     while True:
#         try:
#             user_input = input("\nEnter your tool query or command: ")
            
#             # Check for commands
#             if user_input.lower() in ['exit', 'quit', 'q']:
#                 print("Exiting RAG system. Goodbye!")
#                 break
                
#             elif user_input.lower() == 'help':
#                 print("\nAvailable commands:")
#                 print("  - 'exit' or 'quit': Terminate the program")
#                 print("  - 'add tool': Add new tool data")
#                 print("  - 'reset tools': Reset tool data and rebuild vector store")
#                 print("  - 'help': Show these commands")
#                 continue
                
#             elif user_input.lower() == 'add tool':
#                 try:
#                     print("\nEnter new tool data in JSON format.")
#                     print("Example: [{\"name\": \"NewTool\", \"description\": \"Tool description\", \"api_details\": \"API info\"}]")
#                     print("Enter 'cancel' to abort adding tools.")
                    
#                     json_input = input("\nNew tool data (JSON): ")
#                     if json_input.lower() == 'cancel':
#                         print("Tool addition cancelled.")
#                         continue
                        
#                     new_tools = parse_json_input(json_input)
#                     if new_tools:
#                         # Validate tool structure before adding
#                         valid_tools = []
#                         for tool in new_tools:
#                             if not isinstance(tool, dict):
#                                 print(f"Error: Skipping invalid tool (not an object): {tool}")
#                                 continue
                                
#                             # Ensure the tool has at least a name
#                             if 'name' not in tool:
#                                 print(f"Error: Skipping tool without a name: {tool}")
#                                 continue
                                
#                             # Add default values for missing fields
#                             if 'description' not in tool:
#                                 tool['description'] = "No description provided."
#                             if 'api_details' not in tool:
#                                 tool['api_details'] = "No API details provided."
                                
#                             valid_tools.append(tool)
                        
#                         if valid_tools:
#                             print("\nAdding new tools to the vector store...")
#                             try:
#                                 # Use silent_mode=True to prevent RAG from running query
#                                 run_rag_system("dummy query - adding tools", valid_tools, update_mode=True, silent_mode=True)
#                                 print("\nTools added successfully!")
#                             except Exception as e:
#                                 print(f"\nError adding tools: {str(e)}")
#                                 print("Tools could not be added. Please try again.")
#                         else:
#                             print("\nNo valid tools to add.")
#                 except Exception as e:
#                     print(f"\nUnexpected error during tool addition: {str(e)}")
#                     print("Tool addition failed. Please try again.")
#                 continue
                
#             elif user_input.lower() == 'reset tools':
#                 try:
#                     print("\nEnter new tool data in JSON format to reset the vector store,")
#                     print("or press Enter to reset with the default tool set.")
#                     print("Enter 'cancel' to abort reset.")
                    
#                     json_input = input("\nNew tool data for reset (JSON or press Enter): ")
#                     if json_input.lower() == 'cancel':
#                         print("Reset cancelled.")
#                         continue
                        
#                     if json_input.strip():
#                         reset_tools = parse_json_input(json_input)
#                         if not reset_tools:
#                             continue
#                     else:
#                         reset_tools = tool_data
                        
#                     print("\nResetting vector store with new tool data...")
#                     try:
#                         # Use silent_mode=True to prevent RAG from running query
#                         run_rag_system("dummy query - resetting tools", reset_tools, update_mode=False, silent_mode=True)
#                         print("\nVector store reset successfully!")
#                     except Exception as e:
#                         print(f"\nError resetting vector store: {str(e)}")
#                         print("Reset failed. Previous data may still be available.")
#                 except Exception as e:
#                     print(f"\nUnexpected error during reset operation: {str(e)}")
#                     print("Reset operation failed. Please try again.")
#                 continue
                
#             # Process normal user query
#             try:
#                 print("\nRetrieving and generating answer...\n")
#                 answer = run_rag_system(user_input)
#                 print("\n\nFinal Answer:\n", answer)
#             except Exception as e:
#                 print(f"\nError processing query: {str(e)}")
#                 print("Unable to complete your query. Please try again or use a different query.")
                
#         except KeyboardInterrupt:
#             print("\n\nKeyboard interrupt detected. Exiting...")
#             break
#         except EOFError:
#             print("\n\nEOF detected. Exiting...")
#             break
#         except Exception as e:
#             print(f"\nUnexpected error: {str(e)}")
#             print("The system encountered an error but will continue running.")
#             print("Please try again with a different query or command.")

# if __name__ == "__main__":
#     try:
#         main()
#     except Exception as e:
#         print(f"\n\nCritical error in main program: {str(e)}")
#         print("The program encountered a critical error and must exit.")
#         print("Please check your environment setup, especially Ollama availability.")
#         print("\nTroubleshooting tips:")
#         print("1. Ensure Ollama is running at http://localhost:11434")
#         print("2. Check that required models are available (nomic-embed-text, deepseek-r1:1.5b)")
#         print("3. Verify network connectivity if using remote Ollama instance")
#         print("4. Check system resources (memory, disk space)")
#         print("\nPress Enter to exit...")