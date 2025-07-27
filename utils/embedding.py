import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from .logger import logger

# Load environment variables, specifically for the API key
load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

def get_vector_store(text_chunks: list[str]):
    """
    Generates embeddings for text chunks and stores them in a FAISS vector store.

    Args:
        text_chunks: A list of text chunks.

    Returns:
        A FAISS vector store object.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logger.info("FAISS vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        # This could be due to an invalid API key, network issues, etc.
        raise RuntimeError(f"Could not connect to OpenAI Embedding service: {e}")