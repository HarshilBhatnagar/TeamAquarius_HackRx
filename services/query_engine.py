import asyncio
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from schemas.request import HackRxRequest
from core.cache import document_cache
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.logger import logger

async def process_query(payload: HackRxRequest) -> list[str]: # Changed back to process_query
    document_url = str(payload.documents)
    
    if document_url in document_cache:
        logger.info(f"Cache HIT for document: {document_url}")
        vector_store = document_cache[document_url]
    else:
        logger.info(f"Cache MISS for document: {document_url}. Starting processing...")
        document_text = get_document_text(url=payload.documents)
        text_chunks = get_text_chunks(text=document_text)
        vector_store = get_vector_store(text_chunks=text_chunks)
        document_cache[document_url] = vector_store
        logger.info(f"Stored new vector store in cache for: {document_url}")

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    async def get_answer(question: str) -> str:
        logger.info(f"Processing question: '{question}'")
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={'k': 10}),
            llm=llm
        )
        retrieved_chunks = await retriever.ainvoke(question)
        context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
        generated_answer = await get_llm_answer(context=context, question=question)
        return generated_answer

    logger.info(f"Generating answers for {len(payload.questions)} questions concurrently...")
    tasks = [get_answer(q) for q in payload.questions]
    
    # Use asyncio.gather() to run all tasks and preserve order
    final_answers = await asyncio.gather(*tasks)
    
    logger.info("All answers generated.")
    return final_answers