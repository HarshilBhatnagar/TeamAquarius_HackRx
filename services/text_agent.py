#!/usr/bin/env python3
"""
Text Agent: Handles text-based questions using RAG approach
"""

import asyncio
from typing import List, Dict, Any
from utils.logger import logger
from utils.document_parser import extract_pdf_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.llm import get_llm_answer_simple
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
import pinecone

class TextAgent:
    """
    Text Agent: Processes text content using RAG approach
    """
    
    def __init__(self):
        self.bm25_retriever = None
        self.pinecone_retriever = None
        self.ensemble_retriever = None
        
    async def setup_retrievers(self, text_content: str):
        """
        Setup retrievers for text processing
        """
        try:
            # Create chunks
            chunks = get_text_chunks(text_content)
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Setup BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(documents=documents)
            self.bm25_retriever.k = 30
            
            # Setup Pinecone retriever (if available)
            try:
                # Convert chunks to Document objects for Pinecone
                documents = [Document(page_content=chunk) for chunk in chunks]
                vector_store = get_vector_store(documents)
                # For now, we'll use BM25 only since Pinecone retriever setup is complex
                self.pinecone_retriever = None
            except Exception as e:
                logger.warning(f"Pinecone setup failed: {e}")
                self.pinecone_retriever = None
            
            # Setup ensemble retriever
            if self.pinecone_retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.pinecone_retriever],
                    weights=[0.9, 0.1]
                )
            else:
                self.ensemble_retriever = self.bm25_retriever
                
            logger.info(f"Text Agent: Retrievers setup with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error setting up text retrievers: {e}")
            raise
    
    async def get_answer(self, question: str, document_content: Any) -> str:
        """
        SIMPLIFIED DEBUG VERSION: Get answer for text-based question
        """
        try:
            # Extract text content
            if hasattr(document_content, 'read'):
                content = document_content.read()
                text_content = extract_pdf_text(content)
            else:
                text_content = str(document_content)
            
            # Validate text content
            logger.info(f"Text Agent: Document content length: {len(text_content)} characters")
            if len(text_content) < 1000:
                logger.warning(f"Text Agent: Very short document content ({len(text_content)} chars), may indicate parsing failure")
            logger.info(f"Text Agent: Document content preview: {text_content[:500]}...")
            
            # Setup retrievers if not already done
            if not self.ensemble_retriever:
                await self.setup_retrievers(text_content)
            
            # SIMPLIFIED RETRIEVAL: Direct ensemble retrieval only
            logger.info(f"Text Agent: Starting simplified retrieval for question: '{question}'")
            
            # Get top 15 chunks directly from ensemble retriever
            initial_chunks = await asyncio.to_thread(self.ensemble_retriever.invoke, question)
            logger.info(f"Text Agent: Initial chunks returned: {len(initial_chunks)}")
            
            # Extract chunk content
            context_chunks = [chunk.page_content for chunk in initial_chunks]
            logger.info(f"Text Agent: Context chunks after extraction: {len(context_chunks)}")
            
            # Log first few chunks for debugging
            for i, chunk in enumerate(context_chunks[:3]):
                logger.info(f"Text Agent: Chunk {i+1} preview: {chunk[:200]}...")
            
            # Create simple context
            context = "\n\n---\n\n".join(context_chunks)
            logger.info(f"Text Agent: Final context length: {len(context)} characters")
            
            # Log context preview
            logger.info(f"Text Agent: Context preview: {context[:500]}...")
            
            # Generate answer with simplified prompt
            answer, _ = await get_llm_answer_simple(context, question)
            
            logger.info(f"Text Agent: Answer generated: {answer[:200]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error in text agent: {e}")
            return "The information is not available in the provided context." 