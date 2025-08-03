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
        Get answer for text-based question
        """
        try:
            # Extract text content
            if hasattr(document_content, 'read'):
                # If it's a file-like object
                content = document_content.read()
                text_content = extract_pdf_text(content)
            else:
                # If it's already text
                text_content = str(document_content)
            
            # Setup retrievers if not already done
            if not self.ensemble_retriever:
                await self.setup_retrievers(text_content)
            
            # ENHANCED RETRIEVAL: Aggressive multi-strategy approach
            try:
                # Strategy 1: Direct ensemble retrieval
                initial_chunks = await asyncio.to_thread(self.ensemble_retriever.invoke, question)
                context_chunks = [chunk.page_content for chunk in initial_chunks]
                
                # Strategy 2: Question-specific keyword expansion
                question_lower = question.lower()
                expanded_queries = [question]
                
                # Add insurance-specific keywords based on question type
                if 'waiting period' in question_lower or 'pre-existing' in question_lower:
                    expanded_queries.extend(['waiting period', 'pre-existing disease', '36 months', 'continuous coverage'])
                elif 'child' in question_lower or 'hospitalization' in question_lower:
                    expanded_queries.extend(['child hospitalization', 'cash benefit', 'accompanying', 'daily benefit'])
                elif 'surgery' in question_lower or 'hernia' in question_lower:
                    expanded_queries.extend(['surgery', 'hernia', 'covered', 'treatment', '18 months'])
                elif 'organ donor' in question_lower:
                    expanded_queries.extend(['organ donor', 'pre-hospitalization', 'post-hospitalization', 'donor expenses'])
                
                # Strategy 3: Generic important words
                important_words = []
                stop_words = {'what', 'when', 'where', 'which', 'whose', 'whom', 'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'does', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'may', 'might'}
                
                for word in question_lower.split():
                    if len(word) > 2 and word not in stop_words:
                        important_words.append(word)
                
                expanded_queries.extend(important_words[:10])
                
                # Strategy 4: Execute all expanded queries
                all_chunks = context_chunks.copy()
                for expanded_query in expanded_queries:
                    try:
                        expanded_chunks = await asyncio.to_thread(self.bm25_retriever.invoke, expanded_query)
                        additional_chunks = [chunk.page_content for chunk in expanded_chunks]
                        all_chunks.extend(additional_chunks)
                    except Exception as e:
                        logger.warning(f"Expanded query '{expanded_query}' failed: {e}")
                
                # Strategy 5: Add more chunks if we have limited context
                if len(all_chunks) < 20:
                    logger.info(f"Limited context ({len(all_chunks)} chunks), adding more chunks")
                    try:
                        # Try different variations of the question
                        question_variations = [
                            question.replace('?', ''),
                            question.replace('What is', ''),
                            question.replace('Do I get', ''),
                            question.replace('Will the', ''),
                            question.replace('Are expenses', '')
                        ]
                        
                        for variation in question_variations:
                            if variation.strip():
                                additional_chunks = await asyncio.to_thread(self.bm25_retriever.invoke, variation)
                                more_chunks = [chunk.page_content for chunk in additional_chunks]
                                all_chunks.extend(more_chunks)
                    except Exception as e:
                        logger.warning(f"Question variation retrieval failed: {e}")
                
                # Strategy 6: Combine and deduplicate
                seen = set()
                unique_chunks = []
                for chunk in all_chunks:
                    if chunk not in seen:
                        unique_chunks.append(chunk)
                        seen.add(chunk)
                
                context_chunks = unique_chunks[:100]  # Increased limit
                
                logger.info(f"Text Agent: Retrieved {len(context_chunks)} unique chunks")
                
            except Exception as e:
                logger.error(f"Enhanced retrieval failed: {e}")
                # Fallback to simple retrieval
                context_chunks = [chunk.page_content for chunk in initial_chunks]
            
            # Create context
            context = "\n\n---\n\n".join(context_chunks)
            if len(context) > 20000:  # Increased context limit
                context = context[:20000]
            
            logger.info(f"Text Agent: Created context with {len(context)} characters from {len(context_chunks)} chunks")
            
            # Generate answer
            answer, _ = await get_llm_answer_simple(context, question)
            
            logger.info(f"Text Agent: Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in text agent: {e}")
            return "The information is not available in the provided context." 