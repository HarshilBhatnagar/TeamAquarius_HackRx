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
You are an expert insurance policy analyst with 15+ years of experience. Your task is to score the relevance of text chunks to insurance-related questions with high precision.

**Question Type Analysis:**
1. **Scenario-based Reasoning**: Questions requiring application of policy rules to specific situations
2. **Quantitative Lookups**: Questions seeking specific amounts, percentages, limits, or timeframes
3. **Exclusion Identification**: Questions about what is covered or excluded
4. **Direct Policy Queries**: Simple lookups for policy terms and conditions
5. **Out-of-Domain**: Questions not related to insurance policy

**Scoring Criteria (0-10 scale):**
- **10**: Direct answer with specific policy details, amounts, percentages, exact terms
- **9**: Very relevant with important policy information, coverage details, conditions
- **8**: Highly relevant with policy clauses, benefits, exclusions
- **7**: Relevant with related policy information, general coverage terms
- **6**: Somewhat relevant with insurance concepts, medical terms
- **5**: Marginally relevant with some related information
- **4**: Low relevance, minimal connection to question
- **3**: Very low relevance, barely related
- **2**: Almost irrelevant, weak connection
- **1**: Irrelevant, no connection
- **0**: Completely irrelevant or contradictory

**Insurance Policy Relevance Factors (Weighted):**
- **Direct Policy Terms** (Weight: 3x): Exact policy clauses, conditions, terms
- **Financial Information** (Weight: 2.5x): Amounts, percentages, limits, calculations
- **Time-based Information** (Weight: 2x): Waiting periods, grace periods, timeframes
- **Coverage Details** (Weight: 2x): What's covered, exclusions, benefits
- **Medical Terms** (Weight: 1.5x): Procedures, treatments, conditions
- **Administrative Info** (Weight: 1x): Contact info, procedures, forms

**Question:** {question}

**Chunks to score:**
{chunks}

**Respond with ONLY a valid JSON array of scores, one for each chunk:**
[score1, score2, score3, ...]
"""

CONFIDENCE_PROMPT = """
You are an expert insurance policy analyst. Rate your confidence in the relevance score you just assigned to each chunk.

**Confidence Levels:**
- **High (0.9-1.0)**: Clear, unambiguous relevance or irrelevance
- **Medium (0.6-0.8)**: Reasonably clear but some ambiguity
- **Low (0.3-0.5)**: Unclear relevance, mixed signals
- **Very Low (0.1-0.2)**: Highly ambiguous, could go either way

**Factors affecting confidence:**
- Clarity of policy language
- Specificity of information
- Directness of relevance
- Presence of exact terms from question
- For scenario-based questions: Presence of applicable rules and conditions
- For quantitative questions: Presence of specific numbers and calculations
- For exclusion questions: Clear coverage/exclusion statements

**Question:** {question}

**Chunks with scores:**
{scored_chunks}

**Respond with ONLY a valid JSON array of confidence scores (0.1-1.0):**
[confidence1, confidence2, confidence3, ...]
"""

async def rerank_chunks(chunks: List[str], query: str, top_k: int = 12) -> List[str]:
    """
    Enhanced reranking using LLM-based relevance scoring with confidence metrics.
    Optimized for insurance documents with multi-criteria evaluation.
    Handles complex question types including scenario-based reasoning and quantitative lookups.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Enhanced reranking {len(chunks)} chunks for query: '{query}'")

        # First stage: Relevance scoring
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"Chunk {i+1}: {chunk}\n\n"

        relevance_prompt = RERANK_PROMPT.format(
            question=query,
            chunks=formatted_chunks
        )

        relevance_response = await reranker_client.chat.completions.create(
            messages=[{"role": "user", "content": relevance_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"}
        )

        try:
            scores_text = relevance_response.choices[0].message.content
            scores = json.loads(scores_text)

            if not isinstance(scores, list) or len(scores) != len(chunks):
                logger.warning("Invalid relevance scores format, using fallback")
                return chunks[:top_k]

            # Second stage: Confidence scoring
            scored_chunks_text = ""
            for i, (score, chunk) in enumerate(zip(scores, chunks)):
                scored_chunks_text += f"Chunk {i+1} (Score: {score}): {chunk}\n\n"

            confidence_prompt = CONFIDENCE_PROMPT.format(
                question=query,
                scored_chunks=scored_chunks_text
            )

            confidence_response = await reranker_client.chat.completions.create(
                messages=[{"role": "user", "content": confidence_prompt}],
                model="gpt-4o-mini",
                temperature=0,
                response_format={"type": "json_object"}
            )

            confidence_scores_text = confidence_response.choices[0].message.content
            confidence_scores = json.loads(confidence_scores_text)

            if not isinstance(confidence_scores, list) or len(confidence_scores) != len(chunks):
                logger.warning("Invalid confidence scores format, using relevance scores only")
                confidence_scores = [0.7] * len(chunks)  # Default confidence

            # Combine relevance and confidence scores
            final_scores = []
            for rel_score, conf_score in zip(scores, confidence_scores):
                # Weighted combination: 70% relevance + 30% confidence
                final_score = (rel_score * 0.7) + (conf_score * 10 * 0.3)
                final_scores.append(final_score)

            # Sort by final scores and return top chunks
            scored_chunks = list(zip(final_scores, chunks))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

            logger.info(f"Enhanced reranking completed. Returning top {len(top_chunks)} chunks")
            return top_chunks

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse scores JSON: {e}, using fallback")
            return chunks[:top_k]

    except Exception as e:
        logger.error(f"Error in enhanced LLM reranking: {e}")
        return chunks[:top_k]

async def rerank_chunks_simple(chunks: List[str], query: str, top_k: int = 12) -> List[str]:
    """
    Enhanced simple reranking using advanced keyword matching and semantic analysis.
    Optimized for insurance documents with improved scoring algorithms.
    Handles complex question types and scenario-based reasoning.
    """
    if not chunks:
        return []

    try:
        logger.info(f"Enhanced simple reranking {len(chunks)} chunks for query: '{query}'")

        scored_chunks = []
        query_words = set(query.lower().split())

        # Enhanced insurance keywords with weighted categories
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

        # Question type detection for better scoring
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

        # Detect question type
        for word in query.lower().split():
            if word in question_types:
                question_type = question_types[word]
                break
        
        # Detect scenario-based questions
        if any(indicator in query.lower() for indicator in scenario_indicators):
            is_scenario_based = True
        
        # Detect quantitative questions
        if any(indicator in query.lower() for indicator in quantitative_indicators):
            is_quantitative = True

        for chunk in chunks:
            score = 0
            chunk_words = set(chunk.lower().split())

            # 1. Direct word overlap (highest weight)
            overlap = len(query_words.intersection(chunk_words))
            score += overlap * 4

            # 2. Insurance keyword matching (weighted by category)
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

            # 4. Scenario-based question optimization
            if is_scenario_based:
                scenario_overlap = len(scenario_indicators.intersection(chunk_words))
                score += scenario_overlap * 3
                
                # Bonus for chunks with rules and conditions
                if any(term in chunk.lower() for term in ['rule', 'condition', 'apply', 'based on', 'subject to']):
                    score += 4

            # 5. Quantitative question optimization
            if is_quantitative:
                # Bonus for chunks with numbers and calculations
                if any(char.isdigit() for char in chunk):
                    score += 5
                
                # Bonus for chunks with specific amounts and limits
                if any(term in chunk.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%']):
                    score += 4

            # 6. Chunk length optimization (optimal: 100-400 words)
            chunk_length = len(chunk.split())
            if 100 <= chunk_length <= 400:
                score += 5
            elif 50 <= chunk_length <= 600:
                score += 3
            elif chunk_length < 50:
                score -= 2

            # 7. Numerical information bonus
            if any(char.isdigit() for char in chunk):
                score += 3

            # 8. Policy-specific terms bonus
            policy_terms = ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%', 'days', 'months', 'years', 'clause', 'section']
            if any(term in chunk.lower() for term in policy_terms):
                score += 3

            # 9. Structured content bonus (tables, lists)
            if '|' in chunk or '-' in chunk or any(char in chunk for char in ['•', '▪', '▫', '○', '●']):
                score += 2

            # 10. Question word presence bonus
            question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'does', 'will', 'can', 'is', 'are']
            if any(word in chunk.lower() for word in question_words):
                score += 1

            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for score, chunk in scored_chunks[:top_k]]

        logger.info(f"Enhanced simple reranking completed. Returning top {len(top_chunks)} chunks")
        return top_chunks

    except Exception as e:
        logger.error(f"Error in enhanced simple reranking: {e}")
        return chunks[:top_k]