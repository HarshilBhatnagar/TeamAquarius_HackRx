from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from .logger import logger
from typing import List

def get_text_chunks(text: str) -> List[Document]: # <-- Changed return type hint
    """
    Splits a long text into a list of LangChain Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    # Use create_documents to get Document objects directly, not just strings
    chunks = text_splitter.create_documents([text])
    logger.info(f"Text split into {len(chunks)} Document chunks.")
    return chunks