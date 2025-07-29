import os
from openai import AsyncOpenAI
from utils.logger import logger
from typing import Tuple

try:
    validation_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

VALIDATION_PROMPT = """
You are a fact-checking assistant. Your task is to verify if the given answer is directly supported by the provided context.

**Rules:**
1. The answer must be directly derivable from the context
2. The answer should not contain information not present in the context
3. The answer should not contradict the context
4. If the answer is not supported by the context, respond with "NO"
5. If the answer is well-supported by the context, respond with "YES"

**Context:**
{context}

**Generated Answer:**
{answer}

**Question:**
{question}

**Is this answer directly supported by the context? Respond only with YES or NO:**
"""

async def validate_answer(context: str, answer: str, question: str) -> Tuple[bool, str]:
    """
    Validate if the generated answer is supported by the context.
    
    Args:
        context: The context used to generate the answer
        answer: The generated answer to validate
        question: The original question
    
    Returns:
        Tuple of (is_valid, corrected_answer)
    """
    try:
        formatted_prompt = VALIDATION_PROMPT.format(
            context=context,
            answer=answer,
            question=question
        )
        
        logger.info(f"Validating answer for question: '{question}'")
        
        response = await validation_client.chat.completions.create(
            messages=[{"role": "user", "content": formatted_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=10
        )
        
        validation_result = response.choices[0].message.content.strip().upper()
        is_valid = validation_result == "YES"
        
        if not is_valid:
            logger.warning(f"Answer validation failed for question: '{question}'")
            # Return a safe fallback answer
            corrected_answer = "The information is not available in the provided context."
        else:
            corrected_answer = answer
            
        logger.info(f"Answer validation completed. Valid: {is_valid}")
        return is_valid, corrected_answer
        
    except Exception as e:
        logger.error(f"Error in answer validation: {e}")
        # In case of validation error, return the original answer
        return True, answer 