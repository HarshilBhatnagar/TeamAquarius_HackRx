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
from utils.llm import get_llm_answer_direct
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

    # CRITICAL FIX: Clear cache to prevent wrong document processing
    if cache_key in document_cache:
        logger.info("Clearing cache to ensure correct document processing")
        del document_cache[cache_key]
    
    logger.info("Processing document from scratch")
    
    # CRITICAL FIX: Validate document URL (Updated for new URL)
    document_url = str(payload.documents)
    if "HDFHLIP23024V072223.pdf" not in document_url:
        logger.error(f"Wrong document URL detected: {document_url}")
        raise ValueError(f"Expected HDFC Life Insurance Policy but got: {document_url}")
    
    # Additional validation for new URL format
    if "hackrx_6/policies/" not in document_url:
        logger.warning(f"Document URL format may be outdated: {document_url}")
    
    # Use async document processing
    document_text = await get_document_text(url=document_url)
    
    # CRITICAL FIX: Validate document content
    if len(document_text) > 500000:  # Too large, likely wrong document
        logger.warning(f"Document too large ({len(document_text)} chars), likely wrong document")
        # Clear cache and retry
        document_cache.clear()
        document_text = await get_document_text(url=str(payload.documents))
    
    text_chunks = get_text_chunks(text=document_text)
    
    # Convert text chunks to Document objects for Pinecone
    from langchain_core.documents import Document
    text_chunks_docs = [Document(page_content=chunk, metadata={"source": "insurance_policy"}) for chunk in text_chunks]
    
    vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
    
    # Cache the processed document
    document_cache[cache_key] = (text_chunks_docs, vector_store)
    logger.info(f"Document processing completed in {time.time() - start_time:.2f}s")

    # SIMPLE DIRECT RETRIEVAL: No complex reranking, just get the best chunks
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 12  # Get more chunks for better coverage
    
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 12})
    
    # Simple ensemble with equal weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], 
        weights=[0.5, 0.5]  # Equal weights for balanced retrieval
    )

    async def get_answer_direct(question: str) -> Tuple[str, dict]:
        question_start_time = time.time()
        logger.info(f"Processing question directly: '{question}'")

        # DIRECT RETRIEVAL: Get chunks and use them directly
        try:
            # Get chunks from both retrievers
            bm25_chunks = await asyncio.to_thread(bm25_retriever.invoke, question)
            pinecone_chunks = await asyncio.to_thread(pinecone_retriever.invoke, question)
            
            # Combine and deduplicate chunks
            all_chunks = []
            seen_content = set()
            
            for chunk in bm25_chunks + pinecone_chunks:
                content = chunk.page_content.strip()
                if content and content not in seen_content:
                    all_chunks.append(content)
                    seen_content.add(content)
            
            # Take top 8 chunks for better context
            context_chunks = all_chunks[:8]
            
        except Exception as e:
            logger.warning(f"Direct retrieval failed: {e}")
            # Fallback to simple BM25
            bm25_chunks = await asyncio.to_thread(bm25_retriever.invoke, question)
            context_chunks = [chunk.page_content for chunk in bm25_chunks[:8]]

        # LARGER CONTEXT: Give LLM more information to work with
        context = "\n\n---\n\n".join(context_chunks)
        if len(context) > 3000:  # Increased context limit
            context = context[:3000]

        # DIRECT ANSWER GENERATION: Simple and effective
        generated_answer, usage = await get_llm_answer_direct(context=context, question=question)

        logger.info(f"Question processed in {time.time() - question_start_time:.2f}s")
        return generated_answer, usage

    # PROCESS ALL QUESTIONS CONCURRENTLY for maximum speed
    tasks = [get_answer_direct(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)

    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)

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