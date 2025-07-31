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
from utils.llm_reranker import rerank_chunks, rerank_chunks_simple
from utils.answer_validator import validate_answer
from utils.logger import logger
import hashlib
import time

# Simple in-memory cache for document processing
document_cache = {}
chunk_cache = {}

def get_cache_key(document_url: str) -> str:
    """Generate cache key for document."""
    return hashlib.md5(document_url.encode()).hexdigest()

async def process_query(payload: HackRxRequest, use_reranker: bool = True, use_validation: bool = True, reranker_type: str = "llm") -> Tuple[List[str], int]:
    """
    Optimized RAG pipeline with caching and performance improvements.
    Optimized for <30 second response time while maintaining 75%+ accuracy.
    """
    document_url = str(payload.documents)
    cache_key = get_cache_key(document_url)

    logger.info(f"Processing document: {document_url}")
    start_time = time.time()

    # Check cache for document processing
    if cache_key in document_cache:
        logger.info("Using cached document processing")
        text_chunks_docs, vector_store = document_cache[cache_key]
    else:
        logger.info("Processing document from scratch")
        # Use async document processing - convert HttpUrl to string
        document_text = await get_document_text(url=str(payload.documents))
        text_chunks = get_text_chunks(text=document_text)
        
        # Convert text chunks to Document objects for Pinecone
        from langchain_core.documents import Document
        text_chunks_docs = [Document(page_content=chunk, metadata={"source": "insurance_policy"}) for chunk in text_chunks]
        
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
        
        # Cache the processed document
        document_cache[cache_key] = (text_chunks_docs, vector_store)
        logger.info(f"Document processing completed in {time.time() - start_time:.2f}s")

    # Multi-stage retrieval configuration for insurance policy accuracy
    # Stage 1: Keyword-based retrieval (BM25)
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 30  # Increased for comprehensive retrieval
    
    # Stage 2: Semantic retrieval (Pinecone)
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 30})
    
    # Stage 3: Policy clause specific retrieval
    policy_clause_retriever = create_policy_clause_retriever(text_chunks_docs)
    
    # Ensemble with policy clause focus
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever, policy_clause_retriever], 
        weights=[0.5, 0.3, 0.2]  # Balanced with policy clause emphasis
    )

    async def get_answer_with_optimizations(question: str) -> Tuple[str, dict]:
        question_start_time = time.time()
        logger.info(f"Processing question: '{question}' with optimized pipeline.")

        # No out-of-domain check - let LLM handle all questions
        logger.info(f"Processing question without out-of-domain filtering: {question}")

        # Stage 1: Initial retrieval (15 chunks) - reduced for speed
        initial_chunks = await asyncio.to_thread(ensemble_retriever.invoke, question)
        initial_context_chunks = [chunk.page_content for chunk in initial_chunks]

        # Stage 2: Single reranking (20 → 8 chunks) - increased for accuracy
        if use_reranker and len(initial_context_chunks) > 8:
            logger.info(f"Single reranking {len(initial_context_chunks)} chunks")
            if reranker_type == "llm":
                final_chunks = await rerank_chunks(initial_context_chunks, question, top_k=8)
            else:
                final_chunks = await rerank_chunks_simple(initial_context_chunks, question, top_k=8)
        else:
            final_chunks = initial_context_chunks[:8]

        # Optimized context formatting
        context = "\n\n---\n\n".join(final_chunks)

        # Generate answer with selective validation for speed
        if use_validation and len(payload.questions) <= 2:
            # Only validate for very small question sets to maintain speed
            generated_answer, usage = await get_llm_answer(context=context, question=question)
            is_valid, validated_answer = await validate_answer(context, generated_answer, question)
            if not is_valid:
                logger.warning(f"Answer validation failed, using validated answer")
                return validated_answer, usage
        else:
            # Skip validation for larger question sets to maintain speed
            generated_answer, usage = await get_llm_answer(context=context, question=question)

        logger.info(f"Question processed in {time.time() - question_start_time:.2f}s")
        return generated_answer, usage

    # Process questions with optimized concurrency
    if len(payload.questions) <= 3:
        # For small question sets, process concurrently
        tasks = [get_answer_with_optimizations(q) for q in payload.questions]
        results = await asyncio.gather(*tasks)
    else:
        # For large question sets, process in smaller batches to avoid overwhelming
        batch_size = 2  # Reduced batch size for better performance
        results = []
        for i in range(0, len(payload.questions), batch_size):
            batch = payload.questions[i:i + batch_size]
            batch_tasks = [get_answer_with_optimizations(q) for q in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)

    total_time = time.time() - start_time
    logger.info(f"Optimized pipeline completed in {total_time:.2f}s. Total tokens: {total_tokens}")
    
    # Memory cleanup after processing
    try:
        from utils.embedding import clear_caches
        clear_caches()
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")
    
    return final_answers, total_tokens

async def generate_answer_with_chain_of_thought(context: str, question: str, llm: ChatOpenAI, approach: int = 0) -> Tuple[str, dict, float]:
    """
    Generate answer using chain-of-thought prompting with different approaches.
    """
    
    if approach == 0:
        # Approach 1: Step-by-step analysis
        prompt = create_chain_of_thought_prompt(context, question, "step_by_step")
    elif approach == 1:
        # Approach 2: Policy clause focus
        prompt = create_chain_of_thought_prompt(context, question, "policy_clause")
    else:
        # Approach 3: Calculation and reasoning
        prompt = create_chain_of_thought_prompt(context, question, "calculation")
    
    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        answer = response.content.strip()
        
        # Extract confidence from answer if present
        confidence = extract_confidence_from_answer(answer)
        
        return answer, response.usage, confidence
        
    except Exception as e:
        logger.error(f"Error generating answer with approach {approach}: {e}")
        return "The information is not available in the provided context.", None, 0.3

def create_chain_of_thought_prompt(context: str, question: str, approach: str) -> str:
    """
    Create chain-of-thought prompts for different approaches.
    """
    
    base_prompt = """You are an expert insurance policy analyst. Follow these steps carefully:

**STEP 1: Question Analysis**
- Identify the key policy terms in the question
- Determine what type of information is being requested
- Note any specific amounts, timeframes, or conditions mentioned

**STEP 2: Context Search**
- Search through the provided context for relevant policy clauses
- Look for exact matches and related terms
- Identify multiple relevant sections if available

**STEP 3: Information Extraction**
- Extract specific details (amounts, percentages, time periods, conditions)
- Note any exclusions or limitations
- Identify coverage scope and applicability

**STEP 4: Answer Formation**
- Provide a comprehensive answer based on the extracted information
- Include specific policy references when possible
- Show calculations if required
- End with confidence level (HIGH/MEDIUM/LOW)

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANALYSIS:**
"""

    if approach == "step_by_step":
        return base_prompt.format(context=context, question=question)
    
    elif approach == "policy_clause":
        return base_prompt + f"""
**POLICY CLAUSE FOCUS:**
- Pay special attention to coverage clauses and exclusions
- Look for specific policy terms and definitions
- Check for waiting periods and conditions

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANALYSIS:**
"""
    
    elif approach == "calculation":
        return base_prompt + f"""
**CALCULATION FOCUS:**
- If calculations are needed, show step-by-step reasoning
- Use exact percentages and amounts from the policy
- Apply any limits or caps mentioned in the policy

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANALYSIS:**
"""

def select_best_answer(answers: List[str], confidences: List[float], question: str) -> Tuple[str, float]:
    """
    Select the best answer using self-consistency checking.
    """
    
    # If all answers are similar, use the highest confidence one
    if are_answers_similar(answers):
        best_idx = confidences.index(max(confidences))
        return answers[best_idx], confidences[best_idx]
    
    # If answers differ significantly, use majority voting or highest confidence
    # For now, use the highest confidence answer
    best_idx = confidences.index(max(confidences))
    return answers[best_idx], confidences[best_idx]

def are_answers_similar(answers: List[str]) -> bool:
    """
    Check if answers are semantically similar.
    """
    if len(answers) < 2:
        return True
    
    # Simple similarity check based on key terms
    key_terms = set()
    for answer in answers:
        # Extract key terms (numbers, policy terms, etc.)
        terms = set(re.findall(r'\b\d+%?\b|\b(?:covered|excluded|limit|waiting|period|amount|rupees|lakhs)\b', answer.lower()))
        key_terms.update(terms)
    
    # If answers share most key terms, they're similar
    return len(key_terms) <= 10  # Threshold for similarity

def extract_confidence_from_answer(answer: str) -> float:
    """
    Extract confidence level from answer.
    """
    answer_lower = answer.lower()
    
    if "confidence: high" in answer_lower or "high confidence" in answer_lower:
        return 0.9
    elif "confidence: medium" in answer_lower or "medium confidence" in answer_lower:
        return 0.7
    elif "confidence: low" in answer_lower or "low confidence" in answer_lower:
        return 0.4
    elif "information is not available" in answer_lower:
        return 0.3
    else:
        # Default confidence based on answer characteristics
        if any(char.isdigit() for char in answer):  # Has specific numbers
            return 0.8
        elif len(answer) > 100:  # Detailed answer
            return 0.7
        else:
            return 0.5

async def process_query_fast(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    Fast processing mode with minimal features for maximum speed.
    Use when speed is critical and some accuracy can be sacrificed.
    """
    return await process_query(payload, use_reranker=False, use_validation=False, reranker_type="none")

async def process_query_accurate(payload: HackRxRequest) -> Tuple[List[str], int]:
    """
    High-accuracy processing mode with optimizations for speed.
    This is the default mode for hackathon evaluation.
    Optimized for 75%+ accuracy target with <30 second response time.
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

def create_policy_clause_retriever(documents):
    """
    Create a specialized retriever for policy clauses.
    """
    from langchain_community.retrievers import BM25Retriever
    
    # Filter documents that contain policy clause keywords
    policy_clause_keywords = [
        'multiple policies', 'contribution', 'other insurance', 'policy coordination',
        'claim settlement', 'coverage', 'exclusions', 'waiting period', 'sum insured',
        'co-payment', 'deductible', 'pre-existing', 'morbidity', 'hospitalization'
    ]
    
    policy_clause_docs = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        if any(keyword in content_lower for keyword in policy_clause_keywords):
            policy_clause_docs.append(doc)
    
    if policy_clause_docs:
        retriever = BM25Retriever.from_documents(policy_clause_docs)
        retriever.k = 15
        return retriever
    else:
        # Fallback to regular BM25 if no policy clauses found
        retriever = BM25Retriever.from_documents(documents)
        retriever.k = 15
        return retriever

def check_out_of_domain_fast(question: str) -> bool:
    """
    Fast out-of-domain detection using keyword matching.
    """
    question_lower = question.lower()
    
    # Keywords that indicate out-of-domain questions
    out_of_domain_keywords = [
        'constitution', 'article', 'parliament', 'legal', 'law', 'court',
        'newton', 'physics', 'gravity', 'motion', 'force', 'mass',
        'vehicle', 'motorcycle', 'car', 'engine', 'spark plug',
        'recipe', 'cooking', 'food', 'ingredient',
        'programming', 'code', 'software', 'database', 'python', 'javascript',
        'capital', 'country', 'geography', 'history', 'biography'
    ]
    
    # Check if question contains out-of-domain keywords
    for keyword in out_of_domain_keywords:
        if keyword in question_lower:
            return True
    
    return False