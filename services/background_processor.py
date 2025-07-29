import os
import hashlib
import json
from celery import Celery
from typing import Dict, Any
from utils.document_parser import get_document_text
from utils.chunking import get_text_chunks
from utils.embedding import get_vector_store
from utils.logger import logger

# Initialize Celery
celery_app = Celery(
    'hackrx',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# In-memory storage for job results (in production, use Redis or database)
job_results = {}

@celery_app.task(bind=True)
def process_document_async(self, document_url: str) -> Dict[str, Any]:
    """
    Background task to process document ingestion.
    
    Args:
        document_url: URL of the document to process
    
    Returns:
        Dictionary containing job status and results
    """
    job_id = self.request.id
    
    try:
        # Update job status
        job_results[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting document processing..."
        }
        
        # Step 1: Download and parse document
        self.update_state(
            state="PROGRESS",
            meta={"progress": 25, "message": "Downloading and parsing document..."}
        )
        document_text = get_document_text(url=document_url)
        
        # Step 2: Create text chunks
        self.update_state(
            state="PROGRESS", 
            meta={"progress": 50, "message": "Creating text chunks..."}
        )
        text_chunks_docs = get_text_chunks(text=document_text)
        
        # Step 3: Create vector store
        self.update_state(
            state="PROGRESS",
            meta={"progress": 75, "message": "Creating vector embeddings..."}
        )
        vector_store = get_vector_store(text_chunks_docs=text_chunks_docs)
        
        # Step 4: Generate document hash for caching
        document_hash = hashlib.md5(document_url.encode()).hexdigest()
        
        # Store results
        result = {
            "status": "completed",
            "progress": 100,
            "message": "Document processing completed successfully",
            "document_hash": document_hash,
            "chunk_count": len(text_chunks_docs),
            "vector_store_created": True
        }
        
        job_results[job_id] = result
        
        logger.info(f"Background document processing completed for job {job_id}")
        return result
        
    except Exception as e:
        error_result = {
            "status": "failed",
            "progress": 0,
            "message": f"Document processing failed: {str(e)}",
            "error": str(e)
        }
        job_results[job_id] = error_result
        logger.error(f"Background document processing failed for job {job_id}: {e}")
        raise

def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of a background job.
    
    Args:
        job_id: The job ID to check
    
    Returns:
        Dictionary containing job status and results
    """
    if job_id in job_results:
        return job_results[job_id]
    
    # Check if job exists in Celery
    task_result = celery_app.AsyncResult(job_id)
    
    if task_result.state == 'PENDING':
        return {
            "status": "pending",
            "progress": 0,
            "message": "Job is pending..."
        }
    elif task_result.state == 'PROGRESS':
        return {
            "status": "processing",
            "progress": task_result.info.get('progress', 0),
            "message": task_result.info.get('message', 'Processing...')
        }
    elif task_result.state == 'SUCCESS':
        return task_result.result
    elif task_result.state == 'FAILURE':
        return {
            "status": "failed",
            "progress": 0,
            "message": "Job failed",
            "error": str(task_result.info)
        }
    else:
        return {
            "status": "unknown",
            "progress": 0,
            "message": "Unknown job status"
        } 