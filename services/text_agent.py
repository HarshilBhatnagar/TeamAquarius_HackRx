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
            self.bm25_retriever.k = 50  # Get much more chunks
            
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
            
            # Enhanced retrieval with query expansion
            logger.info(f"Text Agent: Retrieving chunks for question: '{question}'")
            
            # Get chunks for original question
            chunks = await asyncio.to_thread(self.bm25_retriever.invoke, question)
            logger.info(f"Text Agent: Retrieved {len(chunks)} chunks for original question")
            
            # For sum insured questions, get even more chunks
            if 'sum insured' in question_lower or 'maximum' in question_lower:
                # Get additional chunks with different queries
                additional_queries = ['table', 'schedule', 'benefits', 'coverage', 'amount']
                for query in additional_queries:
                    try:
                        extra_chunks = await asyncio.to_thread(self.bm25_retriever.invoke, query)
                        chunks.extend(extra_chunks)
                        logger.info(f"Text Agent: Added {len(extra_chunks)} chunks for '{query}'")
                    except Exception as e:
                        logger.warning(f"Additional query '{query}' failed: {e}")
            
            # Add query expansion for better coverage
            expanded_queries = []
            question_lower = question.lower()
            
            if 'sum insured' in question_lower or 'maximum' in question_lower:
                expanded_queries = [
                    'sum insured', 'coverage amount', 'policy amount', 'maximum coverage',
                    'Rs.', 'rupees', 'amount', 'coverage', 'insured amount',
                    'table', 'schedule', 'benefits', 'coverage details'
                ]
            elif 'eligibility' in question_lower:
                expanded_queries = ['eligibility', 'age', 'entry age', 'minimum age', 'maximum age']
            elif 'policy term' in question_lower:
                expanded_queries = ['policy term', 'duration', 'period', 'years']
            elif 'premium' in question_lower or 'payment' in question_lower:
                expanded_queries = ['premium', 'payment', 'frequency', 'monthly', 'yearly']
            
            # Get additional chunks from expanded queries
            for query in expanded_queries:
                try:
                    additional_chunks = await asyncio.to_thread(self.bm25_retriever.invoke, query)
                    chunks.extend(additional_chunks)
                    logger.info(f"Text Agent: Added {len(additional_chunks)} chunks for query '{query}'")
                except Exception as e:
                    logger.warning(f"Expanded query '{query}' failed: {e}")
            
            # Deduplicate chunks
            seen = set()
            unique_chunks = []
            for chunk in chunks:
                if chunk.page_content not in seen:
                    seen.add(chunk.page_content)
                    unique_chunks.append(chunk)
            
            chunks = unique_chunks
            logger.info(f"Text Agent: Final unique chunks: {len(chunks)}")
            
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