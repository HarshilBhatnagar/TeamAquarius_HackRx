import os
from openai import AsyncOpenAI
from utils.logger import logger
from typing import Tuple

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

try:
    with open("prompts/template.txt", "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    raise FileNotFoundError("Prompt template file not found at 'prompts/template.txt'")

async def get_llm_answer(context: str, question: str) -> Tuple[str, dict]:
    """
    Generates an answer using the specified model and returns the answer and token usage.
    """
    formatted_prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        logger.info(f"Sending request to OpenAI for question: '{question}'")
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            model="gpt-4o-mini", # <-- Updated model name
            temperature=0.1,
        )
        
        answer = chat_completion.choices[0].message.content
        usage = chat_completion.usage
        
        logger.info(f"Received answer. Tokens used: {usage.total_tokens}")
        return answer.strip(), usage
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return "Sorry, there was an error communicating with the language model.", None