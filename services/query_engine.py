import asyncio
import pickle
from langchain_openai import ChatOpenAI
from schemas.request import HackRxRequest
from core.redis_client import redis_client
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.reranker import rerank_documents
from utils.logger import logger

async def process_query(payload: HackRxRequest) -> list[str]:
    document_url = str(payload.documents)
    
    # Use Redis for caching the persistent Pinecone vector store object
    if redis_client and redis_client.exists(document_url):
        logger.info(f"Redis Cache HIT for document: {document_url}")
        # Deserialize the vector store object from Redis
        cached_vs = redis_client.get(document_url)
        vector_store = pickle.loads(cached_vs)
    else:
        logger.info(f"Redis Cache MISS for document: {document_url}. Starting processing...")
        document_text = get_document_text(url=payload.documents)
        text_chunks_docs = get_text_chunks(text=document_text)
        
        # This function now creates and upserts to Pinecone
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
        
        if redis_client:
            # Serialize the vector store object and store it in Redis with a 1-hour TTL
            redis_client.set(document_url, pickle.dumps(vector_store), ex=3600)
            logger.info(f"Stored new Pinecone vector store in Redis cache for: {document_url}")

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    async def get_answer(question: str) -> str:
        logger.info(f"Processing question: '{question}'")

        # 1. Base retriever (Pinecone) fetches a broad set of 25 documents
        base_retriever = vector_store.as_retriever(search_kwargs={'k': 25})
        initial_chunks = await base_retriever.ainvoke(question)
        
        # 2. Reranker precisely re-scores and selects the top 5
        reranked_chunks = rerank_documents(query=question, documents=initial_chunks, top_n=5)
        
        context = "\n\n".join([chunk.page_content for chunk in reranked_chunks])
        
        logger.info(f"Final reranked context for '{question}':\n---\n{context}\n---")
        
        generated_answer = await get_llm_answer(context=context, question=question)
        return generated_answer

    logger.info(f"Generating answers for {len(payload.questions)} questions concurrently...")
    tasks = [get_answer(q) for q in payload.questions]
    
    final_answers = await asyncio.gather(*tasks)
    
    logger.info("All answers generated.")
    return final_answers