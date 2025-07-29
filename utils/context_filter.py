from utils.logger import logger
from utils.llm import client as openai_client

FILTER_PROMPT = """
You are a highly efficient text relevance filter. From the 'Full Context' provided below, your task is to extract and repeat ONLY the exact sentences or data points that are directly relevant to answering the 'User Question'.
If no part of the context is relevant, respond with "No relevant context found.".

== FULL CONTEXT ==
{context}

== USER QUESTION ==
{question}

== RELEVANT CONTEXT ==
"""

async def filter_context_with_llm(question: str, context: str) -> str:
    """
    Uses an LLM to filter a broad context down to only the most relevant sentences.
    """
    prompt = FILTER_PROMPT.format(context=context, question=question)
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini", # Use the fast model for this task
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        filtered_context = response.choices[0].message.content.strip()
        logger.info("Successfully filtered context using LLM.")
        return filtered_context
    except Exception as e:
        logger.error(f"Error during context filtering: {e}. Falling back to original context.")
        return context # Fallback to the original context in case of error