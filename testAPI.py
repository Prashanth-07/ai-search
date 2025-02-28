from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Union
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
from langchain.docstore.document import Document

# Environment and warning setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Tool Recommendation RAG System",
    description="A FastAPI application for AI tool recommendations using RAG",
    version="1.0.0"
)

# Global variable to cache the vector store
VECTOR_STORE = None

# Pydantic models for request/response validation
class Tool(BaseModel):
    name: str
    description: str = Field(default="No description provided.")
    api_details: str = Field(default="No API details provided.")

class ToolList(BaseModel):
    tools: List[Tool]

class Query(BaseModel):
    query: str

class Response(BaseModel):
    response: str
    status: str = "success"
    message: Optional[str] = None

# Utility functions (keeping the same functionality)
def convert_tools_to_documents(tool_data: List[Tool]) -> List[Document]:
    """Converts a list of Tool objects into Document objects."""
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
    """Concatenates Document objects into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def group_documents_by_json_boundary(documents, max_chars=300):
    """Groups Document objects while respecting JSON boundaries."""
    grouped_docs = []
    current_group = ""
    for doc in documents:
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
    """Creates a FAISS vector store from documents."""
    try:
        embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
        sample_vector = embeddings.embed_query("sample text")
        index = faiss.IndexFlatL2(len(sample_vector))
        
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        grouped_docs = group_documents_by_json_boundary(documents, max_chars=300)
        vector_store.add_documents(documents=grouped_docs)
        return vector_store
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up vector store: {str(e)}")

def create_rag_chain(retriever):
    """Creates the RAG chain with the same prompt and configuration."""
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
7. If needed, ask a follow-up question to clarify the user's intent.

Response Format:
- If a single tool is a perfect match:
  - Tool Name: [Brief Description] – Why it's the best match.
- If multiple tools match:
  - Tool 1 Name: [Brief Description] – Why it fits.
  - Tool 2 Name: [Brief Description] – Alternative option.
- If no exact match:
  - "There is no exact match, but here are some alternatives:"
  - Alternative 1 Name: [Description] – How it partially meets the need."""

    model = ChatOllama(
        model="mistral:7b-instruct-q2_K",
        base_url="http://localhost:11434",
        top_p=0.9,
        temperature=0.7,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        stream=False,  # Changed to False for API response
        max_tokens=150
    )
    
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return rag_chain

def get_vector_store(tool_data: Optional[List[Tool]] = None, update_mode: bool = False):
    """Manages the vector store with error handling."""
    global VECTOR_STORE
    try:
        if tool_data and len(tool_data) > 0:
            if VECTOR_STORE is None or not update_mode:
                documents = convert_tools_to_documents(tool_data)
                VECTOR_STORE = setup_vector_store(documents)
            else:
                documents = convert_tools_to_documents(tool_data)
                grouped_docs = group_documents_by_json_boundary(documents, max_chars=300)
                VECTOR_STORE.add_documents(documents=grouped_docs)
        elif VECTOR_STORE is None:
            raise HTTPException(
                status_code=400,
                detail="No vector store available. Initialize with tool data first."
            )
        return VECTOR_STORE
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Vector store error: {str(e)}"
        )

# API Endpoints
@app.post("/initialize", response_model=Response)
async def initialize_system(tool_list: ToolList):
    """Initialize or reset the RAG system with new tool data."""
    try:
        get_vector_store(tool_list.tools, update_mode=False)
        return Response(
            response="Vector store initialized successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-tools", response_model=Response)
async def add_tools(tool_list: ToolList):
    """Add new tools to the existing system."""
    try:
        get_vector_store(tool_list.tools, update_mode=True)
        return Response(
            response="Tools added successfully",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=Response)
async def query_system(query: Query):
    """Process a user query and return tool recommendations."""
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 3, 'lambda_mult': 1.0}
        )
        rag_chain = create_rag_chain(retriever)
        response = rag_chain.invoke(query.query)
        
        return Response(
            response=response,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example tool data for initialization
default_tools = [
    Tool(
        name="VintageImageGen",
        description="Generates vintage-style images.",
        api_details="Endpoint: /api/vintage, Method: POST, Params: {'style': 'vintage'}"
    ),
    Tool(
        name="ModernImageGen",
        description="Generates modern images with high resolution",
        api_details="Endpoint: /api/modern, Method: POST, Params: {'style': 'modern'}"
    )
    # Add more default tools as needed
]

@app.on_event("startup")
async def startup_event():
    """Initialize the system with default tools on startup."""
    try:
        await initialize_system(ToolList(tools=default_tools))
    except Exception as e:
        print(f"Warning: Failed to initialize with default tools: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)