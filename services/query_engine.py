import asyncio
import pickle
from typing import Tuple, List
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from schemas.request import HackRxRequest
from core.redis_client import redis_client
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.context_filter import filter_context_with_llm
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

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 1. Initialize Hybrid Search components to fetch a WIDE set of results
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 20
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 20})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5]
    )

    async def get_answer_with_usage(question: str) -> Tuple[str, dict]:
        logger.info(f"Stage 1: Running Hybrid Search for '{question}'.")
        # 2. Run the synchronous Hybrid Search retriever in a separate thread
        broad_context_chunks = await asyncio.to_thread(ensemble_retriever.invoke, question)
        broad_context = "\n\n".join([chunk.page_content for chunk in broad_context_chunks])

        logger.info(f"Stage 2: Filtering context with LLM for '{question}'.")
        # 3. Use the new LLM-based filter for precision
        precise_context = await filter_context_with_llm(question, broad_context)
        
        logger.info(f"Stage 3: Generating final answer for '{question}'.")
        generated_answer, usage = await get_llm_answer(context=precise_context, question=question)
        return generated_answer, usage

    tasks = [get_answer_with_usage(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)
    
    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)
    
    logger.info(f"All answers generated. Total tokens used: {total_tokens}")
    return final_answers, total_tokens