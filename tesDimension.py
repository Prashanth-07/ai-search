#!/usr/bin/env python
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file (if any)
load_dotenv()

# Initialize the embedding model (adjust model and base_url as needed)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# Embed a sample text to get a sample vector
sample_text = "sample text"
sample_vector = embeddings.embed_query(sample_text)

# Determine the dimension by checking the length of the vector
dimension = len(sample_vector)
print(f"Embedding dimension is: {dimension}")
