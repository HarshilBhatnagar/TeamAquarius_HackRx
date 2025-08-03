from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re
from utils.logger import logger

def get_text_chunks(text: str) -> List[str]:
    """
    Get more chunks to capture more content
    """
    try:
        logger.info(f"Chunking text of length: {len(text)}")
        
        # Early termination for very short texts
        if len(text) < 1000:
            return [text]
        
        # Get more chunks to capture more content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks to get more content
            chunk_overlap=200,  # Good overlap
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
            if len(chunk.strip()) >= 50:  # Lower minimum length
                processed_chunks.append(chunk.strip())
        
        # Get more chunks
        max_chunks = 40
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