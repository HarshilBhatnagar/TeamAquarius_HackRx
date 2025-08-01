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
from utils.llm import get_llm_answer
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

    # Check cache for document processing
    if cache_key in document_cache:
        logger.info("Using cached document processing")
        text_chunks_docs, vector_store = document_cache[cache_key]
    else:
        logger.info("Processing document from scratch")
        # Use async document processing
        document_text = await get_document_text(url=str(payload.documents))
        text_chunks = get_text_chunks(text=document_text)
        
        # Convert text chunks to Document objects for Pinecone
        from langchain_core.documents import Document
        text_chunks_docs = [Document(page_content=chunk, metadata={"source": "insurance_policy"}) for chunk in text_chunks]
        
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
        
        # Cache the processed document
        document_cache[cache_key] = (text_chunks_docs, vector_store)
        logger.info(f"Document processing completed in {time.time() - start_time:.2f}s")

    # INTELLIGENT RETRIEVAL: Optimized for accuracy
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 10  # Increased for better coverage
    
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 10})
    
    # Optimized ensemble with BM25 priority (faster and more accurate for insurance)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], 
        weights=[0.7, 0.3]  # BM25 priority for speed and accuracy
    )

    async def get_answer_agentic(question: str) -> Tuple[str, dict]:
        question_start_time = time.time()
        logger.info(f"Processing question agentically: '{question}'")

        # INTELLIGENT RETRIEVAL: Get chunks with scores for smart ordering
        try:
            # Get chunks with their relevance scores
            initial_chunks = await asyncio.to_thread(ensemble_retriever.invoke, question)
            
            # Extract chunks and their scores (if available)
            chunk_data = []
            for i, chunk in enumerate(initial_chunks[:6]):  # Get 6 chunks for better coverage
                # Estimate relevance score based on position (earlier = more relevant)
                relevance_score = 1.0 - (i * 0.1)  # Simple scoring
                chunk_data.append({
                    'content': chunk.page_content,
                    'score': relevance_score
                })
            
            # INTELLIGENT CONTEXT STRUCTURING: Place most relevant at beginning and end
            if len(chunk_data) >= 3:
                # Sort by relevance score
                chunk_data.sort(key=lambda x: x['score'], reverse=True)
                
                # Place most relevant at beginning and end (combat "Lost in the Middle")
                most_relevant = chunk_data[0]['content']
                second_relevant = chunk_data[1]['content']
                remaining_chunks = [c['content'] for c in chunk_data[2:]]
                
                # Structure: Most relevant -> Others -> Second most relevant
                context_chunks = [most_relevant] + remaining_chunks + [second_relevant]
            else:
                context_chunks = [c['content'] for c in chunk_data]
            
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            context_chunks = ["Error retrieving context"]

        # OPTIMIZED CONTEXT: Limit to 2500 characters for better coverage
        context = "\n\n---\n\n".join(context_chunks)
        if len(context) > 2500:
            context = context[:2500]

        # FAST ANSWER GENERATION: Optimized for speed
        generated_answer, usage = await get_llm_answer(context=context, question=question)

        logger.info(f"Question processed in {time.time() - question_start_time:.2f}s")
        return generated_answer, usage

    # PROCESS ALL QUESTIONS CONCURRENTLY for maximum speed
    tasks = [get_answer_agentic(q) for q in payload.questions]
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