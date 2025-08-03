import os
import hashlib
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from .logger import logger
from typing import List, Dict, Optional

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise EnvironmentError("Pinecone environment variables not set.")

# Chunk-level caching for embeddings
chunk_cache = {}
embedding_cache = {}
MAX_CACHE_SIZE = 100  # Maximum number of cached items
CACHE_CLEANUP_INTERVAL = 300  # Cleanup every 5 minutes
last_cleanup = time.time()

def get_cache_key(chunks: List[Document]) -> str:
    """Generate cache key for chunks."""
    # Create a hash of chunk contents for caching
    chunk_contents = [doc.page_content for doc in chunks]
    content_hash = hashlib.md5(str(chunk_contents).encode()).hexdigest()
    return f"chunks_{content_hash}"

def cleanup_cache():
    """Clean up old cache entries to prevent memory bloat."""
    global last_cleanup, chunk_cache, embedding_cache
    
    current_time = time.time()
    if current_time - last_cleanup > CACHE_CLEANUP_INTERVAL:
        # Clear caches if they get too large
        if len(chunk_cache) > MAX_CACHE_SIZE:
            chunk_cache.clear()
            logger.info("Cleared chunk cache due to size limit")
        
        if len(embedding_cache) > MAX_CACHE_SIZE:
            embedding_cache.clear()
            logger.info("Cleared embedding cache due to size limit")
        
        last_cleanup = current_time

def get_vector_store(text_chunks_docs: List[Document]):
    """
    Creates embeddings from Document objects and upserts them to a Pinecone index.
    Includes chunk-level caching and deduplication.
    """
    try:
        cleanup_cache()  # Periodic cleanup
        
        # Generate cache key
        cache_key = get_cache_key(text_chunks_docs)
        
        # Check cache first
        if cache_key in chunk_cache:
            logger.info("Using cached vector store")
            return chunk_cache[cache_key]
        
        # Deduplicate chunks before processing
        unique_chunks = []
        seen_contents = set()
        
        for doc in text_chunks_docs:
            content_hash = hashlib.md5(doc.page_content.lower().encode()).hexdigest()
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_chunks.append(doc)
        
        if len(unique_chunks) < len(text_chunks_docs):
            logger.info(f"Deduplicated chunks: {len(text_chunks_docs)} -> {len(unique_chunks)}")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store with unique chunks
        pinecone_vs = PineconeVectorStore.from_documents(
            documents=unique_chunks, 
            embedding=embeddings, 
            index_name=PINECONE_INDEX_NAME
        )
        
        # Cache the result
        chunk_cache[cache_key] = pinecone_vs
        
        logger.info(f"Pinecone vector store created/updated for index '{PINECONE_INDEX_NAME}' with {len(unique_chunks)} unique chunks.")
            
        return pinecone_vs
    except Exception as e:
        logger.error(f"Failed to get Pinecone vector store: {e}")
        raise RuntimeError(f"Could not get Pinecone vector store: {e}")

def clear_caches():
    """Clear all caches to free memory."""
    global chunk_cache, embedding_cache
    chunk_cache.clear()
    embedding_cache.clear()
    logger.info("All caches cleared")

def clear_pinecone_index():
    """Clear the entire Pinecone index to remove old/duplicate data."""
    try:
        import pinecone
        from langchain_openai import OpenAIEmbeddings
        
        # Initialize Pinecone
        pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
        
        # Delete all vectors from the index
        index = pinecone.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        
        logger.info(f"Cleared entire Pinecone index '{PINECONE_INDEX_NAME}'")
        return True
    except Exception as e:
        logger.error(f"Failed to clear Pinecone index: {e}")
        return False
    chunk_cache.clear()
    embedding_cache.clear()
    logger.info("All caches cleared")