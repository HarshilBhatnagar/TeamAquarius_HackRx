import os
import json
from typing import List, Tuple
from openai import AsyncOpenAI
from utils.logger import logger

try:
    reranker_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

RERANK_PROMPT = """
You are a relevance-ranking assistant. Your task is to score the relevance of text chunks to a given question.

**Instructions:**
1. For each chunk, assign a relevance score from 0-10
2. 10 = Highly relevant, directly answers the question
3. 5 = Somewhat relevant, contains related information
4. 0 = Not relevant, doesn't help answer the question

**Question:** {question}

**Chunks to score:**
{chunks}

**Respond with ONLY a valid JSON array of scores, one for each chunk:**
[score1, score2, score3, ...]
"""

async def rerank_chunks(chunks: List[str], query: str, top_k: int = 7) -> List[str]:
    """
    Rerank chunks using LLM-based relevance scoring.
    
    Args:
        chunks: List of text chunks to rerank
        query: The user's question
        top_k: Number of top chunks to return
    
    Returns:
        List of top-k most relevant chunks
    """
    if not chunks:
        return []
    
    try:
        logger.info(f"Reranking {len(chunks)} chunks for query: '{query}'")
        
        # Format chunks for scoring
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"Chunk {i+1}: {chunk}\n\n"
        
        # Create scoring prompt
        prompt = RERANK_PROMPT.format(
            question=query,
            chunks=formatted_chunks
        )
        
        # Get relevance scores from LLM
        response = await reranker_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # Parse scores
        try:
            scores_text = response.choices[0].message.content
            scores = json.loads(scores_text)
            
            # Ensure we have the right number of scores
            if isinstance(scores, list) and len(scores) == len(chunks):
                # Create (score, chunk) pairs and sort by score
                scored_chunks = list(zip(scores, chunks))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                
                # Return top-k chunks
                top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
                
                logger.info(f"Successfully reranked chunks. Returning top {len(top_chunks)} chunks")
                return top_chunks
            else:
                logger.warning("Invalid scores format, returning original chunks")
                return chunks[:top_k]
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse scores JSON, returning original chunks")
            return chunks[:top_k]
        
    except Exception as e:
        logger.error(f"Error in LLM reranking: {e}")
        # Fallback to original chunks
        return chunks[:top_k]

async def rerank_chunks_simple(chunks: List[str], query: str, top_k: int = 7) -> List[str]:
    """
    Simple reranking using keyword matching and chunk length optimization.
    
    Args:
        chunks: List of text chunks to rerank
        query: The user's question
        top_k: Number of top chunks to return
    
    Returns:
        List of top-k most relevant chunks
    """
    if not chunks:
        return []
    
    try:
        logger.info(f"Simple reranking {len(chunks)} chunks for query: '{query}'")
        
        # Simple scoring based on keyword overlap and chunk quality
        scored_chunks = []
        query_words = set(query.lower().split())
        
        for chunk in chunks:
            score = 0
            chunk_words = set(chunk.lower().split())
            
            # Keyword overlap score
            overlap = len(query_words.intersection(chunk_words))
            score += overlap * 2
            
            # Chunk length optimization (prefer medium-length chunks)
            chunk_length = len(chunk.split())
            if 50 <= chunk_length <= 200:
                score += 3
            elif 20 <= chunk_length <= 300:
                score += 1
            
            # Bonus for chunks containing question words
            if any(word in chunk.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                score += 1
            
            scored_chunks.append((score, chunk))
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]
        
        logger.info(f"Simple reranking completed. Returning top {len(top_chunks)} chunks")
        return top_chunks
        
    except Exception as e:
        logger.error(f"Error in simple reranking: {e}")
        return chunks[:top_k]