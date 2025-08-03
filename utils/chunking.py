from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re
from utils.logger import logger

def get_text_chunks(text: str) -> List[str]:
    """
    Simple, effective chunking for insurance policy documents.
    """
    try:
        logger.info(f"Chunking text of length: {len(text)}")
        
        # Early termination for very short texts
        if len(text) < 1000:
            return [text]
        
        # Simple chunking for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Medium chunks for good context
            chunk_overlap=300,  # Good overlap for retrieval
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
        
        # Simple post-processing
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= 100:  # Minimum meaningful length
                processed_chunks.append(chunk.strip())
        
        # Limit chunks for performance
        max_chunks = 25
        if len(processed_chunks) > max_chunks:
            logger.info(f"Limiting chunks from {len(processed_chunks)} to {max_chunks}")
            processed_chunks = processed_chunks[:max_chunks]
        
        logger.info(f"Chunking completed: {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in chunking: {e}")
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