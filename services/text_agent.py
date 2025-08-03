#!/usr/bin/env python3
"""
Text Agent: Simple, direct RAG approach
"""

import asyncio
from typing import List, Dict, Any
from utils.logger import logger
from utils.document_parser import extract_pdf_text
from utils.chunking import get_text_chunks
from utils.llm import get_llm_answer_simple
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

class TextAgent:
    """
    Simple Text Agent: Direct RAG approach
    """
    
    def __init__(self):
        self.bm25_retriever = None
        
    async def setup_retrievers(self, text_content: str):
        """
        Setup simple BM25 retriever
        """
        try:
            # Create chunks
            chunks = get_text_chunks(text_content)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Setup BM25 retriever only
            self.bm25_retriever = BM25Retriever.from_documents(documents=documents)
            self.bm25_retriever.k = 25  # Get more chunks
            
            logger.info(f"Text Agent: BM25 retriever setup with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error setting up text retrievers: {e}")
            raise
    
    async def get_answer(self, question: str, document_content: Any) -> str:
        """
        Simple, direct answer generation
        """
        try:
            # Extract text content
            if hasattr(document_content, 'read'):
                content = document_content.read()
                text_content = extract_pdf_text(content)
            else:
                text_content = str(document_content)
            
            logger.info(f"Text Agent: Document content length: {len(text_content)} characters")
            
            # Setup retriever if not already done
            if not self.bm25_retriever:
                await self.setup_retrievers(text_content)
            
            # Simple retrieval: just get top chunks for the question
            logger.info(f"Text Agent: Retrieving chunks for question: '{question}'")
            
            chunks = await asyncio.to_thread(self.bm25_retriever.invoke, question)
            logger.info(f"Text Agent: Retrieved {len(chunks)} chunks")
            
            # Extract chunk content
            context_chunks = [chunk.page_content for chunk in chunks]
            
            # Create context
            context = "\n\n---\n\n".join(context_chunks)
            logger.info(f"Text Agent: Context length: {len(context)} characters")
            
            # Generate answer
            answer, _ = await get_llm_answer_simple(context, question)
            
            logger.info(f"Text Agent: Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in text agent: {e}")
            return "The information is not available in the provided context." 