import asyncio
import pickle
from typing import Tuple, List
from langchain_openai import ChatOpenAI
from schemas.request import HackRxRequest
from core.redis_client import redis_client
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.logger import logger

async def process_query(payload: HackRxRequest) -> Tuple[List[str], int]:
    document_url = str(payload.documents)
    
    if redis_client and redis_client.exists(document_url):
        logger.info(f"Redis Cache HIT for document chunks: {document_url}")
        cached_data = redis_client.get(document_url)
        text_chunks_docs = pickle.loads(cached_data)
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs, from_cache=True)
    else:
        logger.info(f"Redis Cache MISS for document: {document_url}. Starting processing...")
        document_text = get_document_text(url=payload.documents)
        text_chunks_docs = get_text_chunks(text=document_text)
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs, from_cache=False)
        if redis_client:
            redis_client.set(document_url, pickle.dumps(text_chunks_docs), ex=3600)
            logger.info(f"Stored new Document chunks in Redis cache for: {document_url}")

    async def get_answer_with_usage(question: str) -> Tuple[str, dict]:
        logger.info(f"Processing question: '{question}' with Direct MMR Retrieval.")
        
        # Use a single, fast, and diverse retriever (MMR)
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 12, 'fetch_k': 50}
        )
        retrieved_chunks = await retriever.ainvoke(question)
        
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        # Use the powerful GPT-4o for the best reasoning on the final context
        generated_answer, usage = await get_llm_answer(context=context, question=question, model_name="gpt-4o-mini")
        return generated_answer, usage

    tasks = [get_answer_with_usage(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)
    
    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)
    
    logger.info(f"All answers generated. Total tokens used: {total_tokens}")
    return final_answers, total_tokens