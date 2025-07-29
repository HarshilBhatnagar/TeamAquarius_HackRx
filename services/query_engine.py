import asyncio
from typing import Tuple, List, Optional
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from schemas.request import HackRxRequest
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.llm_reranker import rerank_chunks, rerank_chunks_simple
from utils.answer_validator import validate_answer
from utils.logger import logger

async def process_query(payload: HackRxRequest, use_reranker: bool = True, use_validation: bool = True, reranker_type: str = "llm") -> Tuple[List[str], int]:
    """
    Process a query with enhanced accuracy through reranking and validation.
    Optimized for insurance document processing and hackathon evaluation.
    Enhanced for 75%+ accuracy target.
    
    Args:
        payload: The request payload containing documents and questions
        use_reranker: Whether to use reranking for enhanced accuracy
        use_validation: Whether to validate answers against context
        reranker_type: Type of reranking ("llm", "simple", or "none")
    
    Returns:
        Tuple of (answers, total_tokens)
    """
    document_url = str(payload.documents)
    
    logger.info(f"Processing document from scratch: {document_url}")
    document_text = get_document_text(url=payload.documents)
    text_chunks_docs = get_text_chunks(text=document_text)
    vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Enhanced hybrid search with more chunks for better coverage
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 40  # Increased for better coverage
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 40})  # Increased for better coverage
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], weights=[0.6, 0.4]  # Slightly favor BM25 for insurance terms
    )

    async def get_answer_with_enhancements(question: str) -> Tuple[str, dict]:
        logger.info(f"Processing question: '{question}' with Enhanced Hybrid Search.")
        
        # Step 1: Get initial chunks from ensemble retriever
        initial_chunks = await asyncio.to_thread(ensemble_retriever.invoke, question)
        initial_context_chunks = [chunk.page_content for chunk in initial_chunks]
        
        # Step 2: Apply reranking if enabled
        if use_reranker and len(initial_context_chunks) > 10:
            logger.info(f"Applying {reranker_type} reranker to {len(initial_context_chunks)} chunks")
            
            if reranker_type == "llm":
                final_chunks = await rerank_chunks(initial_context_chunks, question, top_k=10)
            elif reranker_type == "simple":
                final_chunks = await rerank_chunks_simple(initial_context_chunks, question, top_k=10)
            else:
                final_chunks = initial_context_chunks[:10]
        else:
            final_chunks = initial_context_chunks[:10]
        
        # Step 3: Create enhanced context with better formatting
        context = "\n\n---\n\n".join(final_chunks)
        
        # Step 4: Generate answer with enhanced prompt
        generated_answer, usage = await get_llm_answer(context=context, question=question)
        
        # Step 5: Validate answer if enabled
        if use_validation:
            is_valid, validated_answer = await validate_answer(context, generated_answer, question)
            if not is_valid:
                logger.warning(f"Answer validation failed for question: '{question}', using fallback")
                return validated_answer, usage
        
        return generated_answer, usage

    tasks = [get_answer_with_enhancements(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)
    
    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)
    
    logger.info(f"All answers generated with enhancements. Total tokens used: {total_tokens}")
    return final_answers, total_tokens

async def process_query_fast(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    Fast processing mode without reranking and validation for speed-critical scenarios.
    
    Args:
        payload: The request payload containing documents and questions
    
    Returns:
        Tuple of (answers, total_tokens)
    """
    return await process_query(payload, use_reranker=False, use_validation=False)

async def process_query_accurate(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    High-accuracy processing mode with both reranking and validation.
    This is the default mode for hackathon evaluation.
    Optimized for 75%+ accuracy target.
    
    Args:
        payload: The request payload containing documents and questions
    
    Returns:
        Tuple of (answers, total_tokens)
    """
    return await process_query(payload, use_reranker=True, use_validation=True, reranker_type="llm")

async def process_query_simple_rerank(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    Processing mode with simple reranking (no LLM calls) and validation.
    
    Args:
        payload: The request payload containing documents and questions
    
    Returns:
        Tuple of (answers, total_tokens)
    """
    return await process_query(payload, use_reranker=True, use_validation=True, reranker_type="simple")