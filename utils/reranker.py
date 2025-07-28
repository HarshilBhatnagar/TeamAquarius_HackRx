from sentence_transformers.cross_encoder import CrossEncoder
from typing import List
from langchain_core.documents import Document
from .logger import logger

try:
    logger.info("Initializing CrossEncoder model for reranking...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    logger.info("CrossEncoder model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize CrossEncoder model: {e}")
    cross_encoder = None

def rerank_documents(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    """
    Reranks documents based on relevance to a query using a CrossEncoder model.
    """
    if not cross_encoder:
        logger.warning("Reranker model not available. Returning original top_n documents.")
        return documents[:top_n]
        
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    
    doc_scores = list(zip(documents, scores))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    reranked_docs = [doc for doc, score in doc_scores[:top_n]]
    logger.info(f"Reranked {len(documents)} documents down to {len(reranked_docs)}.")
    
    return reranked_docs    