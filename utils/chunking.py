from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .logger import logger
from typing import List

def get_text_chunks(text: str) -> List[Document]:
    """
    Splits a long text into a list of LangChain Document objects using a hybrid strategy.
    It uses a general splitter for prose and a table-aware splitter for structured data.
    """
    # 1. General splitter for prose
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    
    # 2. Rule-based splitter for clauses (as suggested in the tips)
    # This helps isolate specific clauses like "Clause X.X..."
    import re
    clauses = re.split(r'(Clause \d+\.\d+)', text)
    
    # Combine and process chunks
    prose_chunks = text_splitter.create_documents(clauses)
    
    # Simple check for table-like structures (can be enhanced)
    # This is a simplified way to handle tables without a full table parser
    final_chunks = []
    for chunk in prose_chunks:
        if "|" in chunk.page_content and "---" in chunk.page_content:
             # This looks like a markdown table, split by rows
            rows = chunk.page_content.split('\n')
            for row in rows:
                if "|" in row and "---" not in row:
                    final_chunks.append(Document(page_content=row))
        else:
            final_chunks.append(chunk)

    logger.info(f"Text split into {len(final_chunks)} final chunks using hybrid strategy.")
    return final_chunks