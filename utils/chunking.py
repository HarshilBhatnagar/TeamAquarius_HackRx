from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from utils.logger import logger

def get_text_chunks(text: str) -> List[str]:
    """
    Optimized text chunking for faster processing.
    Optimized for <30 second response time while maintaining accuracy.
    """
    try:
        logger.info(f"Optimized chunking for text of length: {len(text)}")
        
        # Early termination for very short texts
        if len(text) < 1000:
            return [text]
        
        # Optimized chunk sizes for speed
        chunk_size = 800  # Reduced from 1000 for faster processing
        chunk_overlap = 200  # Reduced from 300 for speed
        
        # Optimized separators for insurance documents
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "? ",    # Questions
            "! ",    # Exclamations
            "; ",    # Semicolons
            ": ",    # Colons
            " - ",   # Dashes
            " | ",   # Pipes (for tables)
            " • ",   # Bullet points
            " ▪ ",   # Square bullets
            " ▫ ",   # White squares
            " ○ ",   # White circles
            " ● ",   # Black circles
            " ",     # Spaces (fallback)
        ]
        
        # Create optimized text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Optimized post-processing for speed
        processed_chunks = []
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.strip()) < 50:
                continue
                
            # Clean up chunk
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                processed_chunks.append(cleaned_chunk)
        
        # Limit total chunks for speed
        max_chunks = 50  # Reduced from unlimited for speed
        if len(processed_chunks) > max_chunks:
            logger.info(f"Limiting chunks from {len(processed_chunks)} to {max_chunks} for speed")
            processed_chunks = processed_chunks[:max_chunks]
        
        logger.info(f"Optimized chunking completed: {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in optimized chunking: {e}")
        # Fallback: simple splitting
        return [text]

def get_dynamic_chunks(text: str) -> List[str]:
    """
    Dynamic chunking based on content type (optimized for speed).
    """
    try:
        logger.info("Using optimized dynamic chunking")
        
        # Quick content type detection
        is_table = "|" in text or "\t" in text
        is_list = any(char in text for char in ['•', '▪', '▫', '○', '●', '-', '*'])
        is_structured = any(term in text.lower() for term in ['clause', 'section', 'chapter', 'part'])
        
        # Optimized chunk sizes based on content type
        if is_table:
            chunk_size = 300  # Smaller for tables
            chunk_overlap = 50
        elif is_list:
            chunk_size = 500  # Medium for lists
            chunk_overlap = 100
        elif is_structured:
            chunk_size = 600  # Medium for structured content
            chunk_overlap = 150
        else:
            chunk_size = 800  # Default for general text
            chunk_overlap = 200
        
        # Use optimized separators
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentences
            "|",     # Table separators
            " • ",   # Bullet points
            " - ",   # Dashes
            " ",     # Spaces
        ]
        
        # Create optimized splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
        
        # Split and process
        chunks = text_splitter.split_text(text)
        
        # Quick post-processing
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) >= 50:  # Minimum length
                processed_chunks.append(chunk.strip())
        
        # Limit chunks for speed
        max_chunks = 40  # Reduced for speed
        if len(processed_chunks) > max_chunks:
            processed_chunks = processed_chunks[:max_chunks]
        
        logger.info(f"Dynamic chunking completed: {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in dynamic chunking: {e}")
        return get_text_chunks(text)  # Fallback to basic chunking