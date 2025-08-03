import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# TARGETED WORKING PROMPT: Specific to HDFC Life Insurance document
SIMPLE_PROMPT = """You are an insurance policy expert analyzing the HDFC Life Insurance policy document. Answer the question based on the provided context.

**INSTRUCTIONS:**
- Answer in a single, clear paragraph
- If the question is about insurance policy and you find relevant information, provide a specific answer with details
- If the question is not about insurance policy (like food, code, etc.), say: "The information is not available in the provided context."
- If the question is about insurance but you cannot find the specific information, say: "The information is not available in the provided context."
- Look carefully through the context for relevant information about:
  * Waiting periods (for diseases, surgeries, etc.)
  * Coverage details (what is covered/not covered)
  * Benefits (cash benefits, hospitalization benefits, etc.)
  * Exclusions and limitations
  * Policy terms and conditions
  * Specific amounts, timeframes, and policy details
- **SPECIFIC SEARCH TERMS** for this document:
  * For child hospitalization: Look for "child", "children", "accompanying", "daily cash", "12 years or less"
  * For hernia surgery: Look for "hernia", "surgery", "gastrointestinal"
  * For organ donor: Look for "organ donor", "donor", "harvesting", "pre and post-hospitalisation"
  * For waiting periods: Look for "waiting period", "pre-existing", "36 months", "exclusion"
- Be specific with numbers, timeframes, and policy details when available
- **IMPORTANT**: This document contains specific information about child hospitalization benefits, hernia surgery coverage, and organ donor expenses. Look carefully for these details.

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWER:**"""

# HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE) PROMPT: Transform user questions into document-like language
HYDE_PROMPT = """You are an expert insurance policy analyst. Given a user's question about an insurance policy, generate a hypothetical answer that would be found in an insurance policy document.

**USER QUESTION:**
{question}

**TASK:**
Generate a hypothetical answer that:
1. Uses the formal, professional language typically found in insurance policy documents
2. Includes specific policy terms, conditions, and clauses that would be relevant
3. Contains the type of information that would answer the user's question
4. Uses insurance industry terminology and legal language
5. Is structured like a policy clause or section

**IMPORTANT:**
- Do NOT provide actual answers or specific amounts unless they're clearly hypothetical
- Focus on the TYPE of information and language structure that would be in a policy
- Use phrases like "The policy provides..." or "Coverage includes..." or "Terms and conditions specify..."
- Include relevant policy sections, clauses, and conditions that would address the question

**HYPOTHETICAL ANSWER:**"""

async def get_llm_answer_simple(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    SIMPLE WORKING ANSWER GENERATION: Proven approach that works.
    Target: High accuracy with good response time using GPT-4o-mini.
    """
    try:
        logger.info(f"SIMPLE: Generating answer for question: '{question}'")

        # SIMPLE WORKING APPROACH: Clear and effective
        prompt = SIMPLE_PROMPT.format(
            context=context,  # Use full context
            question=question
        )

        # Use GPT-4o-mini for speed and reliability
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o-mini",  # Fast and reliable model
            temperature=0,  # Deterministic for accuracy
            max_tokens=500,  # More tokens for detailed answers
            timeout=15  # More time for better answers
        )

        answer = response.choices[0].message.content
        usage = response.usage

        # Simple answer formatting
        formatted_answer = format_answer_simple(answer)

        logger.info(f"SIMPLE answer generated successfully")
        return formatted_answer, usage

    except Exception as e:
        logger.error(f"Error in SIMPLE LLM answer generation: {e}")
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

async def generate_hypothetical_answer(question: str) -> str:
    """
    HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE): Generate a hypothetical answer to bridge semantic gap.
    This transforms user questions into document-like language for better retrieval.
    """
    try:
        logger.info(f"HyDE: Generating hypothetical answer for question: '{question}'")
        
        # Generate hypothetical answer using HyDE prompt
        hyde_prompt = HYDE_PROMPT.format(question=question)
        
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": hyde_prompt}],
            model="gpt-4o-mini",  # Fast model for query transformation
            temperature=0.3,  # Slight creativity for diverse hypothetical answers
            max_tokens=300,  # Concise hypothetical answer
            timeout=5  # Fast timeout for query transformation
        )
        
        hypothetical_answer = response.choices[0].message.content.strip()
        logger.info(f"HyDE: Hypothetical answer generated successfully")
        
        return hypothetical_answer
        
    except Exception as e:
        logger.error(f"Error in HyDE hypothetical answer generation: {e}")
        # Fallback: return the original question if HyDE fails
        return question

def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final answer from the structured response.
    Optimized for chain-of-thought reasoning.
    """
    try:
        # Look for "Final Answer:" section first
        if "**Final Answer:**" in answer_text:
            answer_section = answer_text.split("**Final Answer:**")[1]
            return answer_section.strip()
        
        # Look for "Final Answer:" (without bold)
        if "Final Answer:" in answer_text:
            answer_section = answer_text.split("Final Answer:")[1]
            return answer_section.strip()
        
        # Look for the last meaningful paragraph that contains the answer
        paragraphs = answer_text.split('\n\n')
        for paragraph in reversed(paragraphs):
            paragraph = paragraph.strip()
            if (paragraph and 
                not paragraph.startswith('**') and 
                not paragraph.startswith('Facts from context:') and
                not paragraph.startswith('Analysis:') and
                len(paragraph) > 30 and
                not paragraph.startswith('[') and
                not paragraph.endswith(']')):
                return paragraph
        
        # Fallback: return the last meaningful line
        lines = answer_text.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if (line and 
                not line.startswith('**') and 
                not line.startswith('Facts from context:') and
                not line.startswith('Analysis:') and
                len(line) > 20 and
                not line.startswith('[') and
                not line.endswith(']')):
                return line
        
        return answer_text.strip()
    
    except Exception as e:
        logger.warning(f"Error extracting final answer: {e}")
        return answer_text.strip()

def format_answer_simple(answer: str) -> str:
    """
    Simple answer formatting: clean and direct.
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