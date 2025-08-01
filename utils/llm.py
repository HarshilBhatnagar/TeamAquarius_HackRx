import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# ROUND 2 AGENTIC PROMPT: Let the LLM understand and reason naturally
AGENTIC_PROMPT = """You are an expert insurance policy analyst with deep understanding of insurance documents, policies, and regulations. Your task is to analyze the provided insurance policy context and answer questions accurately.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**AGENTIC ANALYSIS APPROACH:**
1. **Understand the Question**: What is being asked? Is it about coverage, calculations, multiple policies, or general policy information?
2. **Search the Context**: Look for relevant information in the policy document
3. **Reason Through the Answer**: Use your expertise to provide accurate, detailed answers
4. **Handle Edge Cases**: If the question is not related to insurance, politely redirect

**INSTRUCTIONS:**
- Provide comprehensive, accurate answers based on the policy context
- If the question is about multiple policies or contribution, look for relevant clauses
- For calculations, show your reasoning and provide exact amounts
- For coverage questions, check inclusions, exclusions, and conditions
- If the question is not insurance-related, respond with "This question is not related to the insurance policy document provided."
- Be specific, clear, and professional in your responses

**ANSWER:**"""

async def get_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    ROUND 2 AGENTIC ANSWER GENERATION: Let the LLM understand and reason naturally.
    Target: 75%+ accuracy with <30 second response time using GPT-4o-mini.
    """
    try:
        logger.info(f"ROUND 2 AGENTIC: Generating answer for question: '{question}'")

        # AGENTIC APPROACH: Let the LLM handle everything naturally
        enhanced_prompt = AGENTIC_PROMPT.format(
            context=context[:4000],  # Limited context for speed
            question=question
        )

        # Use GPT-4o-mini for faster responses
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": enhanced_prompt}],
            model="gpt-4o-mini",  # Faster model
            temperature=0.1,
            max_tokens=500,  # Reduced for speed
            timeout=8  # Reduced timeout
        )

        answer = response.choices[0].message.content
        usage = response.usage

        # Clean and format answer
        final_answer = extract_final_answer(answer)
        formatted_answer = format_answer_for_sample(final_answer)

        logger.info(f"ROUND 2 agentic answer generated successfully")
        return formatted_answer, usage

    except Exception as e:
        logger.error(f"Error in ROUND 2 agentic LLM answer generation: {e}")
        # Fallback to simple answer generation
        return await get_simple_llm_answer(context, question)

async def get_simple_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Simple fallback answer generation for error cases.
    Optimized for speed with GPT-4o-mini.
    """
    try:
        simple_prompt = f"""
        You are an expert insurance policy analyst. Answer the question based on the provided context.
        If the question is not related to insurance policy, respond with "This question is not related to the insurance policy document provided."

        Context: {context[:2000]}
        Question: {question}

        Answer:
        """

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": simple_prompt}],
            model="gpt-4o-mini",  # Faster model
            temperature=0,
            max_tokens=300,  # Reduced for speed
            timeout=6
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
    Extract the final answer from the response.
    Optimized for quality.
    """
    try:
        # Look for the "Answer:" section
        if "**Answer:**" in answer_text:
            answer_section = answer_text.split("**Answer:**")[1]
            return answer_section.strip()
        
        # Look for numbered steps and extract the final answer
        if "4. " in answer_text:
            lines = answer_text.split('\n')
            for line in reversed(lines):
                if line.strip().startswith('4. ') or line.strip().startswith('Final answer:'):
                    return line.strip()
        
        # Fallback: return the last meaningful paragraph
        paragraphs = answer_text.split('\n\n')
        for paragraph in reversed(paragraphs):
            if paragraph.strip() and not paragraph.startswith('**') and len(paragraph.strip()) > 20:
                return paragraph.strip()
        
        return answer_text.strip()
    
    except Exception as e:
        logger.warning(f"Error extracting final answer: {e}")
        return answer_text.strip()

def format_answer_for_sample(answer: str) -> str:
    """
    Format answer to match the sample response style: clear, concise, specific.
    """
    try:
        # Clean up the answer
        cleaned_answer = answer.strip()
        
        # Remove any markdown formatting
        cleaned_answer = cleaned_answer.replace("**", "").replace("*", "")
        
        # Ensure it ends with proper punctuation
        if not cleaned_answer.endswith(('.', '!', '?')):
            cleaned_answer += '.'
        
        return cleaned_answer
        
    except Exception as e:
        logger.warning(f"Error formatting answer: {e}")
        return answer

# Legacy functions for compatibility (simplified)
def classify_question_type(question: str) -> str:
    """Legacy function - not used in Round 2 agentic approach."""
    return "general"

async def check_out_of_domain_fast(question: str, context: str) -> bool:
    """Legacy function - not used in Round 2 agentic approach."""
    return False

def extract_confidence(answer_text: str) -> str:
    """Legacy function - not used in Round 2 agentic approach."""
    return "Medium"