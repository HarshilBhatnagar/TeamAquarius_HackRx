from langchain.text_splitter import RecursiveCharacterTextSplitter
from .logger import logger

def get_text_chunks(text: str) -> list[str]:
    """
    Splits a long text into chunks optimized for a balance of detail and context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,         # <-- Optimal size for mixed-detail questions
        chunk_overlap=200,        # <-- Increased overlap to preserve context
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_text(text=text)
    logger.info(f"Text split into {len(chunks)} chunks.")
    return chunks