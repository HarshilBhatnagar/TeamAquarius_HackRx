import os
import json
from typing import List, Tuple, Dict
from openai import AsyncOpenAI
from utils.logger import logger

try:
    reranker_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

RERANK_PROMPT = """
You are an expert insurance policy analyst. Score the relevance of text chunks to insurance questions (0-10 scale).

**Scoring:**
- **10**: Direct answer with specific policy details, amounts, percentages
- **9**: Very relevant with important policy information
- **8**: Highly relevant with policy clauses, benefits
- **7**: Relevant with related policy information
- **6**: Somewhat relevant with insurance concepts
- **5**: Marginally relevant with some related information
- **4**: Low relevance, minimal connection
- **3**: Very low relevance, barely related
- **2**: Almost irrelevant, weak connection
- **1**: Irrelevant, no connection
- **0**: Completely irrelevant

**Question:** {question}

**Chunks to score:**
{chunks}

**Respond with ONLY a valid JSON array of scores:**
[score1, score2, score3, ...]
"""

async def rerank_chunks(chunks: List[str], query: str, top_k: int = 8) -> List[str]:
    """
    Optimized reranking using LLM-based relevance scoring.
    Optimized for speed while maintaining accuracy.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Optimized reranking {len(chunks)} chunks for query: '{query}'")

        # Early termination for small chunk sets
        if len(chunks) <= top_k:
            return chunks

        # Limit chunks for faster processing
        max_chunks = min(len(chunks), 20)  # Limit to 20 chunks for speed
        chunks_to_score = chunks[:max_chunks]

        formatted_chunks = ""
        for i, chunk in enumerate(chunks_to_score):
            formatted_chunks += f"Chunk {i+1}: {chunk}\n\n"

        relevance_prompt = RERANK_PROMPT.format(
            question=query,
            chunks=formatted_chunks
        )

        relevance_response = await reranker_client.chat.completions.create(
            messages=[{"role": "user", "content": relevance_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,  # Reduced for speed
            response_format={"type": "json_object"}
        )

        try:
            scores_text = relevance_response.choices[0].message.content
            scores = json.loads(scores_text)

            if not isinstance(scores, list) or len(scores) != len(chunks_to_score):
                logger.warning("Invalid scores format, using fallback")
                return chunks[:top_k]

            # Sort by scores and return top chunks
            scored_chunks = list(zip(scores, chunks_to_score))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

            logger.info(f"Optimized reranking completed. Returning top {len(top_chunks)} chunks")
            return top_chunks

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse scores JSON: {e}, using fallback")
            return chunks[:top_k]

    except Exception as e:
        logger.error(f"Error in optimized LLM reranking: {e}")
        return chunks[:top_k]

async def rerank_chunks_simple(chunks: List[str], query: str, top_k: int = 8) -> List[str]:
    """
    Optimized simple reranking using enhanced keyword matching.
    Optimized for speed with improved scoring algorithms.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Optimized simple reranking {len(chunks)} chunks for query: '{query}'")

        # Early termination for small chunk sets
        if len(chunks) <= top_k:
            return chunks

        scored_chunks = []
        query_words = set(query.lower().split())

        # Optimized insurance keywords with weighted categories
        insurance_keywords = {
            # High priority (weight: 3)
            'coverage', 'policy', 'insured', 'premium', 'claim', 'exclusion', 'limit',
            'waiting', 'period', 'sum', 'medical', 'hospital', 'treatment', 'surgery',
            'disease', 'condition', 'benefit', 'payment', 'grace', 'renewal', 'bonus',
            'discount', 'room', 'icu', 'charges', 'percentage', 'amount', 'rupees',
            'lakhs', 'thousand', 'hundred', 'days', 'months', 'years', 'continuous',
            
            # Medium priority (weight: 2)
            'clause', 'section', 'chapter', 'part', 'article', 'provision', 'term',
            'condition', 'requirement', 'eligibility', 'qualification', 'document',
            'certificate', 'endorsement', 'rider', 'add-on', 'optional', 'mandatory',
            
            # Low priority (weight: 1)
            'insurance', 'health', 'medical', 'surgical', 'diagnostic', 'therapeutic',
            'preventive', 'curative', 'palliative', 'rehabilitative', 'emergency'
        }

        # Quick question type detection
        question_types = {
            'what': ['definition', 'description', 'explanation'],
            'how': ['process', 'procedure', 'method'],
            'when': ['timing', 'schedule', 'deadline'],
            'where': ['location', 'place', 'venue'],
            'who': ['person', 'entity', 'organization'],
            'why': ['reason', 'cause', 'purpose'],
            'which': ['choice', 'option', 'selection'],
            'does': ['coverage', 'inclusion', 'exclusion'],
            'will': ['future', 'prediction', 'outcome'],
            'can': ['possibility', 'capability', 'permission']
        }

        # Scenario-based question indicators
        scenario_indicators = {
            'my', 'i', 'am', 'have', 'bill', 'cost', 'payment', 'co-payment', 'deductible',
            'age', 'years', 'old', 'condition', 'treatment', 'procedure', 'hospitalization'
        }

        # Quantitative question indicators
        quantitative_indicators = {
            'amount', 'limit', 'maximum', 'minimum', 'percentage', 'percent', '%',
            'rupees', 'rs.', 'lakh', 'thousand', 'hundred', 'days', 'months', 'years',
            'how much', 'what is the', 'limit for', 'maximum for'
        }

        question_type = None
        is_scenario_based = False
        is_quantitative = False

        # Quick detection
        for word in query.lower().split():
            if word in question_types:
                question_type = question_types[word]
                break
        
        if any(indicator in query.lower() for indicator in scenario_indicators):
            is_scenario_based = True
        
        if any(indicator in query.lower() for indicator in quantitative_indicators):
            is_quantitative = True

        # Optimized scoring loop
        for chunk in chunks:
            score = 0
            chunk_words = set(chunk.lower().split())

            # 1. Direct word overlap (highest weight)
            overlap = len(query_words.intersection(chunk_words))
            score += overlap * 4

            # 2. Insurance keyword matching (optimized)
            for keyword in chunk_words:
                if keyword in insurance_keywords:
                    if keyword in ['coverage', 'policy', 'insured', 'premium', 'claim']:
                        score += 3
                    elif keyword in ['clause', 'section', 'chapter', 'part']:
                        score += 2
                    else:
                        score += 1

            # 3. Question type optimization
            if question_type:
                type_keywords = set()
                for q_type in question_type:
                    type_keywords.update(q_type.split())
                type_overlap = len(type_keywords.intersection(chunk_words))
                score += type_overlap * 2

            # 4. Scenario-based optimization
            if is_scenario_based:
                scenario_overlap = len(scenario_indicators.intersection(chunk_words))
                score += scenario_overlap * 3
                
                if any(term in chunk.lower() for term in ['rule', 'condition', 'apply', 'based on', 'subject to']):
                    score += 4

            # 5. Quantitative optimization
            if is_quantitative:
                if any(char.isdigit() for char in chunk):
                    score += 5
                
                if any(term in chunk.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%']):
                    score += 4

            # 6. Optimized chunk length scoring
            chunk_length = len(chunk.split())
            if 100 <= chunk_length <= 400:
                score += 5
            elif 50 <= chunk_length <= 600:
                score += 3
            elif chunk_length < 50:
                score -= 2

            # 7. Quick numerical and policy term bonuses
            if any(char.isdigit() for char in chunk):
                score += 3

            policy_terms = ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%', 'days', 'months', 'years', 'clause', 'section']
            if any(term in chunk.lower() for term in policy_terms):
                score += 3

            # 8. Structured content bonus
            if '|' in chunk or '-' in chunk or any(char in chunk for char in ['•', '▪', '▫', '○', '●']):
                score += 2

            scored_chunks.append((score, chunk))

        # Sort and return top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

        logger.info(f"Optimized simple reranking completed. Returning top {len(top_chunks)} chunks")
        return top_chunks

    except Exception as e:
        logger.error(f"Error in optimized simple reranking: {e}")
        return chunks[:top_k]