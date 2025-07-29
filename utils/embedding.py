import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from .logger import logger
from typing import List

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise EnvironmentError("Pinecone environment variables not set.")

def get_vector_store(text_chunks_docs: List[Document], from_cache: bool = False):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if from_cache:
        pinecone_vs = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        logger.info(f"Loaded existing Pinecone index '{PINECONE_INDEX_NAME}' for cached request.")
    else:
        pinecone_vs = PineconeVectorStore.from_documents(
            documents=text_chunks_docs, 
            embedding=embeddings, 
            index_name=PINECONE_INDEX_NAME
        )
        logger.info(f"Pinecone vector store created/updated for index '{PINECONE_INDEX_NAME}'.")
        
    return pinecone_vs