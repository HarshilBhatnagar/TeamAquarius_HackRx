from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re
from utils.logger import logger

def get_text_chunks(text: str) -> List[str]:
    """
    ROUND 2 OPTIMIZED CHUNKING: Fast, simple, and effective.
    Target: Speed and reliability over complexity.
    """
    try:
        logger.info(f"ROUND 2 chunking for text of length: {len(text)}")
        
        # Early termination for very short texts
        if len(text) < 1000:
            return [text]
        
        # WORKING CHUNKING: Proven approach for insurance documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Balanced chunk size
            chunk_overlap=150,  # Good overlap for continuity
            separators=[
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                " ",       # Spaces (fallback)
            ],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_text(text)
        
        # SIMPLE POST-PROCESSING: Filter and limit
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= 100:  # Minimum meaningful length
                processed_chunks.append(chunk.strip())
        
        # WORKING CHUNK LIMIT: Proven approach for insurance documents
        max_chunks = 30
        if len(processed_chunks) > max_chunks:
            logger.info(f"Limiting chunks from {len(processed_chunks)} to {max_chunks}")
            processed_chunks = processed_chunks[:max_chunks]
        
        logger.info(f"ROUND 2 chunking completed: {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in ROUND 2 chunking: {e}")
        # Fallback: simple splitting
        return [text]

# Legacy functions for compatibility
def extract_policy_clauses(text: str) -> List[str]:
    """Legacy function - not used in Round 2."""
    return []

def process_remaining_text(text: str, clause_chunks: List[str]) -> List[str]:
    """Legacy function - not used in Round 2."""
    return get_text_chunks(text)

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """Legacy function - not used in Round 2."""
    return chunks

def get_dynamic_chunks(text: str) -> List[str]:
    """Legacy function - not used in Round 2."""
    return get_text_chunks(text)