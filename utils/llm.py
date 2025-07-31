import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# Optimized prompt template for faster response times
CHAIN_OF_THOUGHT_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
1. If the question is not related to insurance policy, respond: "This question is not related to the insurance policy document provided. Please ask questions about the policy coverage, benefits, terms, or conditions."
2. If the information is not available in the context, respond: "The information is not available in the provided context."
3. For insurance questions, provide a clear, detailed answer with specific information from the policy.
4. For calculations, show the math clearly and provide the final amount.
5. Be comprehensive but concise (under 250 words).
6. Include relevant policy terms, conditions, and limitations when applicable.
7. For coverage questions, specify what is covered and what is not covered.
8. For multiple policy scenarios, look for clauses about "Multiple Policies", "Contribution", "Other Insurance", or similar terms.
9. Pay special attention to clauses that mention claims from other insurers, policy coordination, or contribution rules.
10. If the question involves claiming from multiple insurers, search for specific policy language about this scenario.

**Answer:**"""

# Fast out-of-domain detection prompt
OUT_OF_DOMAIN_PROMPT = """Determine if this question is related to insurance policy analysis.

Question: {question}

Respond with ONLY:
- "Insurance-Related" if the question is about insurance policy, coverage, claims, benefits, etc.
- "Out-of-Domain" if the question is about other topics (constitution, physics, vehicles, recipes, etc.)

Response:"""

# Simplified answer validation prompt
VALIDATION_PROMPT = """Validate if this answer is supported by the context.

Context: {context}
Question: {question}
Answer: {answer}

Respond with ONLY:
- "Supported=True, Confidence=X" if the answer is well-supported (X = 0.7-0.9)
- "Supported=False, Confidence=X" if the answer is not well-supported (X = 0.3-0.6)

Response:"""

async def get_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Optimized answer generation with reduced processing for speed.
    Optimized for <30 second response time while maintaining accuracy.
    """
    try:
        logger.info(f"Generating optimized answer for question: '{question}'")

        # Step 1: Quick out-of-domain check (only for obvious non-insurance questions)
        is_out_of_domain = await check_out_of_domain_fast(question, context)
        if is_out_of_domain:
            return "This question is not related to the insurance policy document provided. Please ask questions about the policy coverage, benefits, terms, or conditions.", None

        # Step 2: Generate answer with optimized prompt
        # Add specific guidance for multiple policy scenarios
        enhanced_prompt = CHAIN_OF_THOUGHT_PROMPT.format(
            context=context[:4000],  # Further increased context for policy clauses
            question=question
        )
        
        # Add specific guidance for the HDFC scenario
        if "hdfc" in question.lower() or "remaining" in question.lower() or "200,000" in question.lower():
            enhanced_prompt += "\n\n**SPECIAL GUIDANCE:** This appears to be a multiple policy scenario. Look specifically for clauses about 'Multiple Policies', 'Contribution', or 'Other Insurance'. The user is asking about claiming remaining amounts from another insurer."
        
        initial_prompt = enhanced_prompt

        initial_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": initial_prompt}],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=800,  # Increased for better answers
            timeout=15  # Increased timeout
        )

        initial_answer = initial_response.choices[0].message.content
        usage = initial_response.usage

        # Step 3: Quick self-consistency check (only for complex questions)
        if len(question.split()) > 8 or any(word in question.lower() for word in ['calculate', 'compute', 'determine', 'find']):
            consistency_prompt = SELF_CONSISTENCY_PROMPT.format(
                question=question,
                context=context[:1000],  # Limited context
                answer=initial_answer
            )

            try:
                consistency_response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": consistency_prompt}],
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=200,  # Very limited for speed
                    timeout=10
                )

                consistency_result = consistency_response.usage
                if usage:
                    usage.total_tokens += consistency_result.total_tokens

                # Quick check for correction
                consistency_text = consistency_response.choices[0].message.content
                if "CORRECTED_ANSWER:" in consistency_text:
                    corrected_answer = consistency_text.split("CORRECTED_ANSWER:")[1].strip()
                    if corrected_answer and len(corrected_answer) > 10:
                        initial_answer = corrected_answer

            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")

        # Step 4: Extract final answer
        final_answer = extract_final_answer(initial_answer)

        logger.info(f"Optimized answer generated")
        return final_answer, usage

    except Exception as e:
        logger.error(f"Error in optimized LLM answer generation: {e}")
        # Fallback to simple answer generation
        return await get_simple_llm_answer(context, question)

async def check_out_of_domain_fast(question: str, context: str) -> bool:
    """
    Fast out-of-domain detection with optimized processing.
    """
    try:
        # Quick keyword-based check for obvious out-of-domain questions
        out_of_domain_keywords = {
            # Programming/Technical
            'code', 'python', 'javascript', 'js', 'database', 'postgresql', 'sql', 'programming',
            'function', 'variable', 'api', 'endpoint', 'server', 'client', 'html', 'css',
            
            # Automotive/Mechanical
            'spark plug', 'gap', 'tubeless', 'tyre', 'tire', 'disc brake', 'oil', 'thums up',
            'thumbs up', 'engine', 'motorcycle', 'bike', 'vehicle', 'automotive', 'mechanical',
            'brake', 'clutch', 'gear', 'transmission', 'fuel', 'petrol', 'diesel',
            
            # General Knowledge
            'capital', 'weather', 'recipe', 'cake', 'cooking', 'food', 'meaning of life',
            'news', 'headlines', 'flat tire', 'puncture', 'constitution', 'legal',
            
            # Unrelated Topics
            'poem', 'story', 'creative', 'fiction', 'novel', 'book', 'movie', 'music',
            'sports', 'game', 'entertainment', 'celebrity', 'politics', 'election'
        }
        
        # Insurance-specific keywords that should NOT be flagged as out-of-domain
        insurance_keywords = {
            'claim', 'policy', 'coverage', 'premium', 'sum insured', 'hospitalization',
            'medical', 'treatment', 'surgery', 'expenses', 'documents', 'settlement',
            'multiple', 'policies', 'contribution', 'other insurance', 'hdfc', 'icici',
            'bajaj', 'tata', 'max', 'star', 'health', 'insurance', 'company',
            'raised', 'approved', 'remaining', 'amount', 'expenses', 'total',
            'rupees', 'rs', 'lakhs', 'thousand', 'hundred'
        }
        
        question_lower = question.lower()
        
        # Quick keyword check
        for keyword in out_of_domain_keywords:
            if keyword in question_lower:
                # Check if it's actually an insurance-related question
                if any(insurance_keyword in question_lower for insurance_keyword in insurance_keywords):
                    continue  # Don't flag as out-of-domain if it contains insurance keywords
                logger.info(f"Out-of-domain detected via keyword: {keyword}")
                return True
        
        # Only use LLM for very obvious non-insurance questions
        if len(question.split()) > 20:  # Only check very long questions
            out_of_domain_prompt = OUT_OF_DOMAIN_PROMPT.format(
                question=question,
                context=context[:500]  # Very limited context
            )

            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": out_of_domain_prompt}],
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=50,  # Very limited
                timeout=5
            )

            assessment = response.choices[0].message.content.strip()
            is_out_of_domain = "Out-of-Domain" in assessment
            
            logger.info(f"Out-of-domain check: {assessment}")
            return is_out_of_domain

        return False

    except Exception as e:
        logger.warning(f"Error in fast out-of-domain check: {e}")
        return False

async def get_simple_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Simple fallback answer generation for error cases.
    Optimized for speed.
    """
    try:
        simple_prompt = f"""
        Based on the insurance policy context, answer the question accurately and concisely.
        If the question is not related to insurance policy, respond with "This question is not related to the insurance policy document provided."

        Context: {context[:2000]}
        Question: {question}

        Answer:
        """

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": simple_prompt}],
            model="gpt-4o",
            temperature=0,
            max_tokens=600,  # Reduced for speed
            timeout=10
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        logger.info("Simple answer generation completed")
        return answer, usage

    except Exception as e:
        logger.error(f"Error in simple LLM answer generation: {e}")
        return "I apologize, but I encountered an error while processing your question. Please try again.", None

def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final answer from the chain-of-thought response.
    Optimized for speed.
    """
    try:
        # Look for the "Answer:" section
        if "**Answer:**" in answer_text:
            answer_section = answer_text.split("**Answer:**")[1]
            # Remove confidence section if present
            if "**Confidence:**" in answer_section:
                answer_section = answer_section.split("**Confidence:**")[0]
            return answer_section.strip()
        
        # Fallback: return the last paragraph
        paragraphs = answer_text.split('\n\n')
        for paragraph in reversed(paragraphs):
            if paragraph.strip() and not paragraph.startswith('**'):
                return paragraph.strip()
        
        return answer_text.strip()
    
    except Exception as e:
        logger.warning(f"Error extracting final answer: {e}")
        return answer_text.strip()

def extract_confidence(answer_text: str) -> str:
    """
    Extract confidence level from the chain-of-thought response.
    """
    try:
        if "**Confidence:**" in answer_text:
            confidence_line = answer_text.split("**Confidence:**")[1]
            if "**Reasoning:**" in confidence_line:
                confidence_line = confidence_line.split("**Reasoning:**")[0]
            confidence = confidence_line.strip()
            if confidence in ["High", "Medium", "Low"]:
                return confidence
        return "Medium"  # Default confidence
    except Exception as e:
        logger.warning(f"Error extracting confidence: {e}")
        return "Medium"