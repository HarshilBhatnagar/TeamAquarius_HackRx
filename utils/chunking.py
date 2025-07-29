from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .logger import logger
from typing import List
import re

def get_text_chunks(text: str) -> List[Document]:
    """
    Splits a long text into a list of LangChain Document objects using a hybrid strategy.
    Optimized for insurance policy documents to preserve policy clauses and details.
    """
    # Enhanced chunking for insurance documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for better context preservation
        chunk_overlap=300,  # Increased overlap for better continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separators for policy documents
    )
    
    # Rule-based splitter for insurance policy clauses
    # This helps isolate specific clauses and sections
    policy_sections = re.split(r'(Clause \d+\.\d+|Section \d+|Chapter \d+|Part \d+)', text)
    
    # Combine and process chunks
    prose_chunks = text_splitter.create_documents(policy_sections)
    
    # Enhanced processing for insurance-specific content
    final_chunks = []
    for chunk in prose_chunks:
        content = chunk.page_content.strip()
        
        # Skip very short chunks that don't contain meaningful information
        if len(content) < 50:
            continue
            
        # Handle table-like structures in policy documents
        if "|" in content and "---" in content:
            # This looks like a markdown table, split by rows
            rows = content.split('\n')
            for row in rows:
                if "|" in row and "---" not in row and len(row.strip()) > 20:
                    final_chunks.append(Document(page_content=row.strip()))
        else:
            final_chunks.append(chunk)

    # Post-processing: Merge very short chunks with adjacent ones
    merged_chunks = []
    i = 0
    while i < len(final_chunks):
        current_chunk = final_chunks[i].page_content
        
        # If chunk is too short, try to merge with next chunk
        if len(current_chunk) < 100 and i + 1 < len(final_chunks):
            next_chunk = final_chunks[i + 1].page_content
            merged_content = current_chunk + "\n\n" + next_chunk
            
            # Only merge if total length is reasonable
            if len(merged_content) <= 1500:
                merged_chunks.append(Document(page_content=merged_content))
                i += 2  # Skip next chunk since we merged it
            else:
                merged_chunks.append(final_chunks[i])
                i += 1
        else:
            merged_chunks.append(final_chunks[i])
            i += 1

    logger.info(f"Text split into {len(merged_chunks)} final chunks optimized for insurance documents.")
    return merged_chunks