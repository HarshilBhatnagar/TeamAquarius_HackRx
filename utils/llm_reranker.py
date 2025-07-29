import json
from typing import List
from langchain_core.documents import Document
from utils.logger import logger
from utils.llm import client as openai_client # Reuse the OpenAI client

async def rerank_with_llm(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    """
    Reranks documents using an LLM to judge relevance.
    """
    if not documents:
        return []

    # Format the passages with indices for the LLM to reference
    formatted_passages = ""
    for i, doc in enumerate(documents):
        formatted_passages += f"Passage {i}: {doc.page_content}\n\n"

    # --- FIX: Use a robust f-string to build the prompt ---
    # This directly and correctly embeds the formatted_passages variable.
    prompt = f'''
You are a relevance-ranking assistant. Based on the user's question and the provided list of numbered text passages, your task is to identify the top 5 most relevant passages for answering the question.

Respond with ONLY a valid JSON object containing a single key "ranked_indices" which is a list of the integer indices of the top 5 most relevant passages, ordered from most to least relevant.

Example Response:
{{"ranked_indices": [3, 12, 1, 8, 22]}}

User Question: "{query}"

Numbered Passages:
{formatted_passages}
'''
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result_json = json.loads(response.choices[0].message.content)
        ranked_indices = result_json.get("ranked_indices", [])
        
        # Select and order the documents based on the LLM's ranking
        reranked_docs = [documents[i] for i in ranked_indices if i < len(documents)]
        
        logger.info(f"LLM Reranked {len(documents)} documents down to {len(reranked_docs)}.")
        return reranked_docs[:top_n]

    except Exception as e:
        logger.error(f"Error during LLM Reranking: {e}. Falling back to top_n.")
        # Fallback in case of an error
        return documents[:top_n]