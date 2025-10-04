from dotenv import load_dotenv
import os
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Load .env file if it exists
load_dotenv()

def get_embedding_function():
    """Return an embedding function using Ollama embeddings."""
    
    print("✅ Using Ollama embeddings")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    return embeddings
