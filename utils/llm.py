import os
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

try:
    with open("prompts/template.txt", "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    raise FileNotFoundError("Prompt template file not found at 'prompts/template.txt'")

async def get_llm_answer(context: str, question: str) -> str:
    """
    Generates an answer using gpt-4o-mini based on the provided context and question.
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
            model="gpt-4o-mini", # <-- Changed to the new model
            temperature=0.1,
        )
        
        usage = chat_completion.usage
        logger.info(f"OpenAI token usage: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}")

        answer = chat_completion.choices[0].message.content
        logger.info(f"Received answer from OpenAI.")
        return answer.strip()
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        return "Sorry, there was an error communicating with the language model."