import asyncio
import re
from typing import Tuple, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from schemas.request import HackRxRequest
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer_simple
from utils.logger import logger
import hashlib
import time

# Simple in-memory cache for document processing
document_cache = {}

def get_cache_key(document_url: str) -> str:
    """Generate cache key for document."""
    return hashlib.md5(document_url.encode()).hexdigest()

async def process_query(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    ROUND 2 AGENTIC PIPELINE: Let the LLM understand and reason naturally.
    Target: 75%+ accuracy, <30 seconds response time using GPT-4o-mini.
    """
    document_url = str(payload.documents)
    cache_key = get_cache_key(document_url)

    logger.info(f"ROUND 2 AGENTIC: Processing document: {document_url}")
    start_time = time.time()

    # CRITICAL FIX: Clear cache and Pinecone to prevent wrong document processing
    if cache_key in document_cache:
        logger.info("Clearing cache to ensure correct document processing")
        del document_cache[cache_key]
    
    # Clear Pinecone index to remove old/duplicate data
    try:
        from utils.embedding import clear_pinecone_index
        if clear_pinecone_index():
            logger.info("Successfully cleared Pinecone index")
        else:
            logger.warning("Failed to clear Pinecone index, continuing anyway")
    except Exception as e:
        logger.warning(f"Error clearing Pinecone index: {e}")
    
    logger.info("Processing document from scratch")
    
    # Process any document URL - removed hardcoded validation
    document_url = str(payload.documents)
    logger.info(f"Processing document: {document_url}")
    
    # Use async document processing
    document_text = await get_document_text(url=document_url)
    
    # Process document content without size restrictions
    logger.info(f"Document content length: {len(document_text)} characters")
    
    text_chunks = get_text_chunks(text=document_text)
    
    # Convert text chunks to Document objects for Pinecone
    from langchain_core.documents import Document
    text_chunks_docs = [Document(page_content=chunk, metadata={"source": "insurance_policy"}) for chunk in text_chunks]
    
    vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
    
    # Cache the processed document
    document_cache[cache_key] = (text_chunks_docs, vector_store)
    logger.info(f"Document processing completed in {time.time() - start_time:.2f}s")

    # ENHANCED RETRIEVAL: Get maximum chunks for comprehensive coverage
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 30  # Get maximum chunks
    
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 30})
    
    # Use BM25 primarily as it works better for insurance documents
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], 
        weights=[0.9, 0.1]  # Heavy BM25 priority for insurance docs
    )

    async def get_answer_simple(question: str) -> Tuple[str, dict]:
        question_start_time = time.time()
        logger.info(f"Master-Slave Architecture: Processing question: '{question}'")

        # MASTER-SLAVE ARCHITECTURE: Use Master Agent to orchestrate Text and Table agents
        try:
            # Initialize Master Agent
            from services.master_agent import MasterAgent
            master_agent = MasterAgent()
            
            # Process question through master agent with the processed document data
            answer = await master_agent.process_question(question, document_text)
            
            logger.info(f"Master-Slave Architecture: Answer generated successfully")
            return answer, {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}  # Placeholder for token count
            
        except Exception as e:
            logger.error(f"Error in master-slave architecture: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again.", {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    # PROCESS ALL QUESTIONS CONCURRENTLY for maximum speed
    tasks = [get_answer_simple(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)

    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].get('total_tokens', 0) for res in results if res[1] is not None)

    total_time = time.time() - start_time
    logger.info(f"ROUND 2 agentic pipeline completed in {total_time:.2f}s. Total tokens: {total_tokens}")
    
    return final_answers, total_tokens

# Legacy functions for compatibility
async def process_query_fast(payload: HackRxRequest) -> Tuple[List[str], int]:
    """Fast processing mode - same as main function now."""
    return await process_query(payload)

async def process_query_accurate(payload: HackRxRequest) -> Tuple[List[str], int]:
    """High-accuracy processing mode - same as main function now."""
    return await process_query(payload)

async def process_query_simple_rerank(payload: HackRxRequest) -> Tuple[List[str], int]:
    """Simple rerank mode - same as main function now."""
    return await process_query(payload)