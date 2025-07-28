import asyncio
import pickle
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from schemas.request import HackRxRequest
from core.redis_client import redis_client
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.logger import logger

async def process_query(payload: HackRxRequest) -> list[str]:
    document_url = str(payload.documents)
    
    if redis_client and redis_client.exists(document_url):
        logger.info(f"Redis Cache HIT for document: {document_url}")
        cached_vs = redis_client.get(document_url)
        vector_store = pickle.loads(cached_vs)
    else:
        logger.info(f"Redis Cache MISS for document: {document_url}. Starting processing...")
        document_text = get_document_text(url=payload.documents)
        text_chunks_docs = get_text_chunks(text=document_text)
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
        
        if redis_client:
            redis_client.set(document_url, pickle.dumps(vector_store), ex=3600)
            logger.info(f"Stored new Pinecone vector store in Redis cache for: {document_url}")

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    async def get_answer(question: str) -> str:
        logger.info(f"Processing question: '{question}' with MultiQueryRetriever.")

        # Use the MultiQueryRetriever directly. This is much more resource-efficient.
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={'k': 10}),
            llm=llm
        )
        retrieved_chunks = await retriever.ainvoke(question)
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        
        logger.info(f"Final context for '{question}':\n---\n{context}\n---")
        
        generated_answer = await get_llm_answer(context=context, question=question)
        return generated_answer

    tasks = [get_answer(q) for q in payload.questions]
    final_answers = await asyncio.gather(*tasks)
    
    return final_answers