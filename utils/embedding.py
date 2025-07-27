import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma # <-- Change import from FAISS to Chroma
from utils.logger import logger

load_dotenv()
if os.getenv("OPENAI_API_KEY") is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

def get_vector_store(text_chunks: list[str]):
    """
    Generates embeddings and stores them in a Chroma vector store.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # Use Chroma instead of FAISS
        vector_store = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
        logger.info("Chroma vector store created successfully.")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise RuntimeError(f"Could not create vector store: {e}")