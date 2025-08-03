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
    Policy-aware LLM-based reranking of chunks for relevance to the query.
    Optimized for insurance policy accuracy.
    """
    if len(chunks) <= top_k:
        return chunks
    
    try:
        logger.info(f"Policy-aware reranking {len(chunks)} chunks for query: '{query}'")
        
        # Classify question type for specialized reranking
        question_type = classify_question_for_reranking(query)
        logger.info(f"Question classified as: {question_type}")
        
        # Create specialized reranking prompt
        prompt = create_specialized_rerank_prompt(query, chunks, question_type)
        
        # Get LLM response
        response = await reranker_llm.ainvoke([{"role": "user", "content": prompt}])
        response_text = response.content.strip()
        
        # Extract scores from response
        scores = extract_scores_from_response(response_text, len(chunks))
        
        if not scores or len(scores) != len(chunks):
            logger.warning(f"Failed to extract valid scores, using simple reranking")
            return rerank_chunks_simple(chunks, query, top_k)
        
        # Apply policy-specific scoring adjustments
        adjusted_scores = apply_policy_scoring_adjustments(chunks, scores, question_type)
        
        # Sort chunks by adjusted scores and return top_k
        chunk_score_pairs = list(zip(chunks, adjusted_scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_chunks = [chunk for chunk, score in chunk_score_pairs[:top_k]]
        logger.info(f"Policy-aware LLM reranking completed: {len(reranked_chunks)} chunks selected")
        
        return reranked_chunks
        
    except Exception as e:
        logger.error(f"Error in policy-aware LLM reranking: {e}")
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

def classify_question_for_reranking(question: str) -> str:
    """
    Classify question type for specialized reranking.
    """
    question_lower = question.lower()
    
    # Multiple policy indicators
    if any(keyword in question_lower for keyword in ['hdfc', 'icici', 'bajaj', 'tata', 'max', 'star', 'allianz', 'bupa', 'multiple', 'policies', 'remaining', 'balance', 'disallowed']):
        return "multiple_policy"
    
    # Coverage indicators
    if any(keyword in question_lower for keyword in ['covered', 'coverage', 'excluded', 'exclusion', 'surgery', 'treatment', 'procedure']):
        return "coverage"
    
    # Calculation indicators
    if any(keyword in question_lower for keyword in ['calculate', 'compute', 'determine', 'how much', 'amount', 'percentage']):
        return "calculation"
    
    return "general"

def create_specialized_rerank_prompt(question: str, chunks: List[str], question_type: str) -> str:
    """
    Create specialized reranking prompt based on question type.
    """
    base_prompt = f"""You are an insurance policy expert. Rate the relevance of each text chunk to the query on a scale of 1-10 (10 being most relevant).

Query: "{question}"

Text chunks:
{chr(10).join(f"{i+1}. {chunk[:200]}..." for i, chunk in enumerate(chunks))}"""

    if question_type == "multiple_policy":
        base_prompt += """

SPECIAL INSTRUCTIONS FOR MULTIPLE POLICY QUESTIONS:
- Give HIGHER scores (8-10) to chunks containing "Multiple Policies", "Contribution", "Other Insurance", "Policy Coordination"
- Give MEDIUM scores (5-7) to chunks about claim settlement, coverage, or policy terms
- Give LOWER scores (1-4) to chunks about unrelated topics
- Focus on clauses that mention multiple insurers or claim coordination"""

    elif question_type == "coverage":
        base_prompt += """

SPECIAL INSTRUCTIONS FOR COVERAGE QUESTIONS:
- Give HIGHER scores (8-10) to chunks containing coverage details, inclusions, exclusions
- Give MEDIUM scores (5-7) to chunks about policy terms, conditions, limits
- Give LOWER scores (1-4) to chunks about unrelated topics
- Focus on specific treatment/procedure coverage information"""

    elif question_type == "calculation":
        base_prompt += """

SPECIAL INSTRUCTIONS FOR CALCULATION QUESTIONS:
- Give HIGHER scores (8-10) to chunks containing amounts, percentages, limits, calculations
- Give MEDIUM scores (5-7) to chunks about policy terms, coverage details
- Give LOWER scores (1-4) to chunks about unrelated topics
- Focus on numerical information and calculation rules"""

    base_prompt += "\n\nReturn ONLY a JSON array of scores [score1, score2, ...] where each score is 1-10."
    
    return base_prompt

def apply_policy_scoring_adjustments(chunks: List[str], scores: List[int], question_type: str) -> List[int]:
    """
    Apply policy-specific scoring adjustments.
    """
    adjusted_scores = scores.copy()
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        
        if question_type == "multiple_policy":
            # Boost scores for multiple policy related content
            if any(term in chunk_lower for term in ['multiple policies', 'contribution', 'other insurance', 'policy coordination']):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 2)
            elif any(term in chunk_lower for term in ['claim', 'settlement', 'coverage']):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 1)
        
        elif question_type == "coverage":
            # Boost scores for coverage related content
            if any(term in chunk_lower for term in ['covered', 'coverage', 'excluded', 'exclusion', 'included']):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 2)
            elif any(term in chunk_lower for term in ['surgery', 'treatment', 'procedure', 'medical']):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 1)
        
        elif question_type == "calculation":
            # Boost scores for calculation related content
            if any(term in chunk_lower for term in ['rupees', 'rs', 'lakhs', 'thousand', 'hundred', 'percentage', 'limit', 'maximum']):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 2)
            elif any(char.isdigit() for char in chunk):
                adjusted_scores[i] = min(10, adjusted_scores[i] + 1)
    
    return adjusted_scores

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