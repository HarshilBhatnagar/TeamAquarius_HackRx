import asyncio
import json
import re
from typing import List, Tuple
from langchain_openai import ChatOpenAI
from utils.logger import logger

# Initialize LLM for reranking
reranker_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=200,
    timeout=30
)

async def rerank_chunks(chunks: List[str], query: str, top_k: int = 8) -> List[str]:
    """
    LLM-based reranking of chunks for relevance to the query.
    Optimized for speed and accuracy.
    """
    if len(chunks) <= top_k:
        return chunks
    
    try:
        logger.info(f"Optimized reranking {len(chunks)} chunks for query: '{query}'")
        
        # Create a simplified prompt for faster processing
        prompt = f"""You are a document retrieval expert. Rate the relevance of each text chunk to the query on a scale of 1-10 (10 being most relevant).

Query: "{query}"

Text chunks:
{chr(10).join(f"{i+1}. {chunk[:200]}..." for i, chunk in enumerate(chunks))}

Return ONLY a JSON array of scores [score1, score2, ...] where each score is 1-10.
Example: [8, 3, 9, 5, 7, 2, 6, 4]"""

        # Get LLM response
        response = await reranker_llm.ainvoke([{"role": "user", "content": prompt}])
        response_text = response.content.strip()
        
        # Extract scores from response
        scores = extract_scores_from_response(response_text, len(chunks))
        
        if not scores or len(scores) != len(chunks):
            logger.warning(f"Failed to extract valid scores, using simple reranking")
            return rerank_chunks_simple(chunks, query, top_k)
        
        # Sort chunks by scores and return top_k
        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks = [chunk for chunk, score in chunk_score_pairs[:top_k]]
        logger.info(f"LLM reranking completed: {len(reranked_chunks)} chunks selected")
        
        return reranked_chunks
        
    except Exception as e:
        logger.error(f"Error in optimized LLM reranking: {e}")
        # Fallback to simple reranking
        return rerank_chunks_simple(chunks, query, top_k)

def extract_scores_from_response(response_text: str, expected_count: int) -> List[int]:
    """
    Extract scores from LLM response with robust parsing.
    """
    try:
        # Try to find JSON array in the response
        json_match = re.search(r'\[[\d,\s]+\]', response_text)
        if json_match:
            scores_json = json_match.group()
            scores = json.loads(scores_json)
            
            # Validate scores
            if isinstance(scores, list) and len(scores) == expected_count:
                # Ensure all scores are integers 1-10
                valid_scores = []
                for score in scores:
                    try:
                        score_int = int(score)
                        if 1 <= score_int <= 10:
                            valid_scores.append(score_int)
                        else:
                            valid_scores.append(5)  # Default score
                    except (ValueError, TypeError):
                        valid_scores.append(5)  # Default score
                
                return valid_scores
        
        # Fallback: try to extract numbers from text
        numbers = re.findall(r'\b([1-9]|10)\b', response_text)
        if len(numbers) >= expected_count:
            scores = [int(num) for num in numbers[:expected_count]]
            return scores
        
        # Last resort: return default scores
        logger.warning(f"Could not extract scores from response: {response_text[:100]}...")
        return [5] * expected_count
        
    except Exception as e:
        logger.error(f"Error extracting scores: {e}")
        return [5] * expected_count

async def rerank_chunks_simple(chunks: List[str], query: str, top_k: int = 8) -> List[str]:
    """
    Simple keyword-based reranking as fallback.
    """
    if len(chunks) <= top_k:
        return chunks
    
    try:
        logger.info(f"Using simple keyword-based reranking for {len(chunks)} chunks")
        
        # Extract key terms from query
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Score chunks based on term overlap
        chunk_scores = []
        for chunk in chunks:
            chunk_terms = set(re.findall(r'\b\w+\b', chunk.lower()))
            overlap = len(query_terms.intersection(chunk_terms))
            score = overlap / max(len(query_terms), 1)  # Normalize by query length
            chunk_scores.append((chunk, score))
        
        # Sort by score and return top_k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        reranked_chunks = [chunk for chunk, score in chunk_scores[:top_k]]
        
        logger.info(f"Simple reranking completed: {len(reranked_chunks)} chunks selected")
        return reranked_chunks
        
    except Exception as e:
        logger.error(f"Error in simple reranking: {e}")
        return chunks[:top_k]