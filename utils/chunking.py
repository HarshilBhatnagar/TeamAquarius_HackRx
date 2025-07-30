from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .logger import logger
from typing import List
import re

def get_text_chunks(text: str) -> List[Document]:
    """
    Dynamic chunking strategy optimized for insurance policy documents.
    Uses adaptive chunk sizes based on content type and semantic boundaries.
    """
    
    # Pre-process text to identify different content types
    sections = preprocess_insurance_document(text)
    
    final_chunks = []
    
    for section_type, content in sections:
        if section_type == "table":
            # Small chunks for tables and structured data
            chunks = create_table_chunks(content)
        elif section_type == "clause":
            # Medium chunks for policy clauses
            chunks = create_clause_chunks(content)
        elif section_type == "list":
            # Small chunks for lists and bullet points
            chunks = create_list_chunks(content)
        else:
            # Large chunks for general policy text
            chunks = create_general_chunks(content)
        
        final_chunks.extend(chunks)
    
    # Post-processing: Merge very short chunks and optimize
    final_chunks = post_process_chunks(final_chunks)
    
    logger.info(f"Dynamic chunking created {len(final_chunks)} optimized chunks for insurance documents.")
    return final_chunks

def preprocess_insurance_document(text: str) -> List[tuple]:
    """
    Pre-process document to identify different content types.
    """
    sections = []
    
    # Split by major sections
    major_sections = re.split(r'(Clause \d+\.\d+|Section \d+|Chapter \d+|Part \d+|Schedule \d+)', text)
    
    for i in range(0, len(major_sections), 2):
        if i + 1 < len(major_sections):
            header = major_sections[i]
            content = major_sections[i + 1]
            
            # Classify content type
            if is_table_content(content):
                sections.append(("table", header + content))
            elif is_list_content(content):
                sections.append(("list", header + content))
            elif is_clause_content(content):
                sections.append(("clause", header + content))
            else:
                sections.append(("general", header + content))
        else:
            # Handle remaining content
            content = major_sections[i]
            if content.strip():
                sections.append(("general", content))
    
    return sections

def is_table_content(content: str) -> bool:
    """Check if content is table-like."""
    return "|" in content and ("---" in content or any(line.count("|") > 2 for line in content.split("\n")))

def is_list_content(content: str) -> bool:
    """Check if content is list-like."""
    lines = content.split("\n")
    list_indicators = sum(1 for line in lines if re.match(r'^[\s]*[•\-\*\d+\.]', line.strip()))
    return list_indicators > len(lines) * 0.3

def is_clause_content(content: str) -> bool:
    """Check if content is clause-like."""
    clause_indicators = ["shall", "will", "must", "should", "covered", "excluded", "subject to", "provided that"]
    return any(indicator in content.lower() for indicator in clause_indicators)

def create_table_chunks(content: str) -> List[Document]:
    """Create small chunks for table content."""
    chunks = []
    
    # Split by rows
    rows = content.split("\n")
    current_chunk = []
    
    for row in rows:
        if "|" in row and "---" not in row and len(row.strip()) > 10:
            current_chunk.append(row.strip())
            
            # Create chunk every 3-5 rows
            if len(current_chunk) >= 4:
                chunks.append(Document(page_content="\n".join(current_chunk)))
                current_chunk = []
    
    # Add remaining rows
    if current_chunk:
        chunks.append(Document(page_content="\n".join(current_chunk)))
    
    return chunks

def create_list_chunks(content: str) -> List[Document]:
    """Create small chunks for list content."""
    chunks = []
    
    # Split by list items
    items = re.split(r'[\n\r]+[\s]*[•\-\*\d+\.]', content)
    current_chunk = []
    
    for item in items:
        if item.strip():
            current_chunk.append(item.strip())
            
            # Create chunk every 5-7 items
            if len(current_chunk) >= 6:
                chunks.append(Document(page_content="\n".join(current_chunk)))
                current_chunk = []
    
    # Add remaining items
    if current_chunk:
        chunks.append(Document(page_content="\n".join(current_chunk)))
    
    return chunks

def create_clause_chunks(content: str) -> List[Document]:
    """Create medium chunks for clause content."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,  # Medium size for clauses
        chunk_overlap=400,  # Good overlap for clause continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return text_splitter.create_documents([content])

def create_general_chunks(content: str) -> List[Document]:
    """Create large chunks for general policy text."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,  # Large size for general text
        chunk_overlap=600,  # Good overlap for context continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return text_splitter.create_documents([content])

def post_process_chunks(chunks: List[Document]) -> List[Document]:
    """Post-process chunks to optimize for retrieval."""
    processed_chunks = []
    i = 0
    
    while i < len(chunks):
        current_chunk = chunks[i].page_content
        
        # If chunk is too short, try to merge with next chunk
        if len(current_chunk) < 150 and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].page_content
            merged_content = current_chunk + "\n\n" + next_chunk
            
            # Only merge if total length is reasonable
            if len(merged_content) <= 2000:
                processed_chunks.append(Document(page_content=merged_content))
                i += 2  # Skip next chunk since we merged it
            else:
                processed_chunks.append(chunks[i])
                i += 1
        else:
            processed_chunks.append(chunks[i])
            i += 1
    
    return processed_chunks