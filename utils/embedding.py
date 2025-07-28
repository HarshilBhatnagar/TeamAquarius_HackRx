import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from .logger import logger
from typing import List

load_dotenv()

# Check for Pinecone credentials
if os.getenv("PINECONE_API_KEY") is None or os.getenv("PINECONE_INDEX_NAME") is None:
    raise EnvironmentError("Pinecone environment variables not set.")

def get_vector_store(text_chunks_docs: List[Document]):
    """
    Generates embeddings and upserts them to a Pinecone index.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        # from_documents will create embeddings and upsert them to the specified Pinecone index
        pinecone_vs = PineconeVectorStore.from_documents(
            documents=text_chunks_docs, 
            embedding=embeddings, 
            index_name=index_name
        )
        logger.info(f"Pinecone vector store created/updated for index '{index_name}'.")
        return pinecone_vs
    except Exception as e:
        logger.error(f"Failed to create Pinecone vector store: {e}")
        raise RuntimeError(f"Could not create Pinecone vector store: {e}")