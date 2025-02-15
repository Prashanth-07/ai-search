#!/usr/bin/env python
import os
import warnings
from dotenv import load_dotenv

# Import LangChain components
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.docstore.document import Document
from pinecone import Pinecone as PineconeClient

# Environment and warning setup
warnings.filterwarnings("ignore")
load_dotenv()

# Pinecone configuration
PINECONE_INDEX_NAME = "ai-tool-search"
VECTOR_STORE = None

# Initialize Pinecone client
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

def convert_tools_to_documents(tool_data):
    """
    Converts a list of dictionaries representing tool data into Document objects.
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

def setup_vector_store(documents):
    """
    Creates a Pinecone vector store from a list of documents using Ollama embeddings.
    """
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url="http://localhost:11434"
    )
    
    # Get texts from documents
    texts = [doc.page_content for doc in documents]
    
    # Create vector store
    return Pinecone.from_texts(
        texts=texts,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

def create_rag_chain(retriever):
    """
    Creates a RAG chain with a prompt template and ChatOllama model.
    """
    prompt = """
    You are an assistant that selects the best matching AI tool based on user requirements.
    Use the following tool information to answer the query.
    If no tool is relevant, state that no matching tool was found.
    
    Question: {question}
    Tool Information: {context}
    
    Answer in bullet points:
    """
    model = ChatOllama(model="deepseek-r1:7b", base_url="http://localhost:11434")
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
    Returns the Pinecone vector store. If tool_data is provided,
    it rebuilds the vector store with the new data.
    """
    global VECTOR_STORE
    if tool_data and len(tool_data) > 0:
        documents = convert_tools_to_documents(tool_data)
        VECTOR_STORE = setup_vector_store(documents)
    elif VECTOR_STORE is None:
        embeddings = OllamaEmbeddings(
            model='nomic-embed-text',
            base_url="http://localhost:11434"
        )
        VECTOR_STORE = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
    return VECTOR_STORE

def verify_embedding_dimensions():
    """
    Verify the dimension of embeddings from nomic-embed-text model.
    """
    embeddings = OllamaEmbeddings(
        model='nomic-embed-text',
        base_url="http://localhost:11434"
    )
    test_embedding = embeddings.embed_query("test")
    print(f"Embedding dimensions: {len(test_embedding)}")
    return len(test_embedding)

def run_rag_system(user_query, tool_data=None):
    """
    Runs the RAG system with Pinecone vector store.
    """
    vector_store = get_vector_store(tool_data)
    retriever = vector_store.as_retriever(search_kwargs={'k': 3})
    rag_chain = create_rag_chain(retriever)
    
    final_response = ""
    for chunk in rag_chain.stream(user_query):
        final_response += chunk
        print(chunk, end="", flush=True)
    return final_response

def main():
    # Verify embedding dimensions
    dims = verify_embedding_dimensions()
    print(f"Confirmed: nomic-embed-text produces {dims}-dimensional embeddings\n")
    # tool_data =[]
    
    # Sample tool data
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
        }
    ]
    
    # Run queries
    user_query = input("Enter your tool query: ")
    print("\nRetrieving and generating answer...\n")
    answer = run_rag_system(user_query, tool_data)
    print("\n\nFinal Answer:\n", answer)
    
    user_query2 = input("\nEnter another query: ")
    print("\nRetrieving and generating answer...\n")
    answer2 = run_rag_system(user_query2)
    print("\n\nFinal Answer:\n", answer2)

if __name__ == "__main__":
    main()