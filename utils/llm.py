import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# ENHANCED PROMPT FOR BETTER ACCURACY
SIMPLE_PROMPT = """You are an expert insurance policy analyst. Answer the question based ONLY on the provided context.

**INSTRUCTIONS:**
- Answer in a single, clear paragraph
- Use ONLY information from the provided context
- Be specific with numbers, time periods, and policy terms
- Look carefully through ALL the context for relevant information
- For tables, read the data carefully and extract specific values
- If you find ANY relevant information, use it to provide an answer
- If the information is not in the context, say: "The information is not available in the provided context."
- If the question is not about insurance policy, say: "The information is not available in the provided context."

**IMPORTANT:**
- Search for keywords related to the question (sum insured, eligibility, policy term, premium, etc.)
- Look in tables for numerical data and policy details
- Check for policy terms, conditions, and coverage amounts
- Be thorough in your search through the context

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWER:**"""

async def get_llm_answer_simple(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Simple, direct answer generation for maximum accuracy.
    """
    try:
        logger.info(f"Generating answer for question: '{question}'")

        prompt = SIMPLE_PROMPT.format(
            context=context,
            question=question
        )

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=300,
            timeout=10
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        logger.info(f"Answer generated successfully")
        return answer, usage

    except Exception as e:
        logger.error(f"Error in LLM answer generation: {e}")
        return "The information is not available in the provided context.", None

def format_answer_simple(answer: str) -> str:
    """Simple answer formatting."""
    try:
        cleaned_answer = answer.strip()
        if not cleaned_answer.endswith(('.', '!', '?')):
            cleaned_answer += '.'
        return cleaned_answer
    except Exception as e:
        logger.warning(f"Error formatting answer: {e}")
        return answer

# Legacy functions for compatibility
def classify_question_type(question: str) -> str:
    return "general"

async def check_out_of_domain_fast(question: str, context: str) -> bool:
    return False

def extract_confidence(answer_text: str) -> str:
    return "Medium"