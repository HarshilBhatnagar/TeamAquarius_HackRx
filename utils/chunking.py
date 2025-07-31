from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import re
from utils.logger import logger

def get_text_chunks(text: str) -> List[str]:
    """
    Clause-based text chunking for insurance policy documents.
    Optimized for preserving complete policy clauses and sections.
    """
    try:
        logger.info(f"Clause-based chunking for text of length: {len(text)}")
        
        # Early termination for very short texts
        if len(text) < 1000:
            return [text]
        
        # First pass: Identify and preserve policy clauses
        clause_chunks = extract_policy_clauses(text)
        
        # Second pass: Process remaining text with clause-aware chunking
        remaining_chunks = process_remaining_text(text, clause_chunks)
        
        # Combine and deduplicate
        all_chunks = clause_chunks + remaining_chunks
        processed_chunks = deduplicate_chunks(all_chunks)
        
        # Limit total chunks for performance
        max_chunks = 80  # Increased to preserve more policy content
        if len(processed_chunks) > max_chunks:
            logger.info(f"Limiting chunks from {len(processed_chunks)} to {max_chunks}")
            processed_chunks = processed_chunks[:max_chunks]
        
        logger.info(f"Clause-based chunking completed: {len(processed_chunks)} chunks")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error in clause-based chunking: {e}")
        # Fallback: simple splitting
        return [text]

def extract_policy_clauses(text: str) -> List[str]:
    """
    Extract complete policy clauses from text.
    """
    clauses = []
    
    # Policy clause patterns
    clause_patterns = [
        r'Multiple Policies[^.]*\.',
        r'Contribution[^.]*\.',
        r'Other Insurance[^.]*\.',
        r'Policy Coordination[^.]*\.',
        r'Claim Settlement[^.]*\.',
        r'Coverage[^.]*\.',
        r'Exclusions[^.]*\.',
        r'Waiting Period[^.]*\.',
        r'Sum Insured[^.]*\.',
        r'Co-payment[^.]*\.',
        r'Deductible[^.]*\.',
        r'Pre-existing[^.]*\.',
        r'Hospitalization[^.]*\.',
        r'Pre-hospitalization[^.]*\.',
        r'Post-hospitalization[^.]*\.'
    ]
    
    for pattern in clause_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            if len(match.strip()) > 50:  # Minimum clause length
                clauses.append(match.strip())
    
    return clauses

def process_remaining_text(text: str, clause_chunks: List[str]) -> List[str]:
    """
    Process remaining text with clause-aware chunking.
    """
    # Remove already extracted clauses from text
    remaining_text = text
    for clause in clause_chunks:
        remaining_text = remaining_text.replace(clause, "")
    
    # Use clause-aware separators
    separators = [
        "\n\n\n",  # Major section breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentences
        "? ",      # Questions
        "! ",      # Exclamations
        "; ",      # Semicolons
        ": ",      # Colons
        " - ",     # Dashes
        " | ",     # Pipes (for tables)
        " • ",     # Bullet points
        " ▪ ",     # Square bullets
        " ▫ ",     # White squares
        " ○ ",     # White circles
        " ● ",     # Black circles
        " ",       # Spaces (fallback)
    ]
    
    # Create clause-aware text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Smaller chunks for remaining text
        chunk_overlap=150,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_text(remaining_text)
    
    # Filter and clean chunks
    processed_chunks = []
    for chunk in chunks:
        if len(chunk.strip()) >= 50:
            processed_chunks.append(chunk.strip())
    
    return processed_chunks

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Remove duplicate chunks while preserving order.
    """
    seen_chunks = set()
    unique_chunks = []
    
    for chunk in chunks:
        chunk_hash = hash(chunk.lower())
        if chunk_hash not in seen_chunks:
            seen_chunks.add(chunk_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks

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