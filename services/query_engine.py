import asyncio
from typing import Tuple, List
from langchain_openai import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from schemas.request import HackRxRequest
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer
from utils.logger import logger

async def process_query(payload: HackRxRequest) -> Tuple[List[str], int]:
    document_url = str(payload.documents)
    
    logger.info(f"Processing document from scratch: {document_url}")
    document_text = get_document_text(url=payload.documents)
    text_chunks_docs = get_text_chunks(text=document_text)
    vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # 1. Initialize BM25 retriever for fast keyword search
    bm25_retriever = BM25Retriever.from_documents(documents=text_chunks_docs)
    bm25_retriever.k = 15

    # 2. Initialize Pinecone retriever for semantic search
    pinecone_retriever = vector_store.as_retriever(search_kwargs={'k': 15})

    # 3. Initialize Ensemble Retriever to combine and weight both methods
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever], weights=[0.5, 0.5]
    )

    async def get_answer_with_usage(question: str) -> Tuple[str, dict]:
        logger.info(f"Processing question: '{question}' with Hybrid Search.")
        
        # Run the synchronous retriever in a separate thread
        initial_chunks = await asyncio.to_thread(ensemble_retriever.invoke, question)
        
        context = "\n\n".join([chunk.page_content for chunk in initial_chunks])
        
        generated_answer, usage = await get_llm_answer(context=context, question=question)
        return generated_answer, usage

    tasks = [get_answer_with_usage(q) for q in payload.questions]
    results = await asyncio.gather(*tasks)
    
    final_answers = [res[0] for res in results]
    total_tokens = sum(res[1].total_tokens for res in results if res[1] is not None)
    
    logger.info(f"All answers generated. Total tokens used: {total_tokens}")
    return final_answers, total_tokens