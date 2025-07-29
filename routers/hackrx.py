from fastapi import APIRouter, Request, Response, status, HTTPException, Depends
from schemas.request import HackRxRequest
from schemas.response import HackRxResponse
from services.query_engine import process_query_accurate
from utils.logger import logger
from utils.security import validate_token

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=HackRxResponse, 
    status_code=status.HTTP_200_OK,
    summary="Run the RAG pipeline for insurance document processing - Hackathon Endpoint",
    dependencies=[Depends(validate_token)]
)
async def run_hackrx(payload: HackRxRequest, request: Request, response: Response):
    """
    Main endpoint for hackathon evaluation.
    Processes insurance documents and answers questions with maximum accuracy.
    """
    ip_address = request.client.host
    logger.info(f"Hackathon evaluation request from IP: {ip_address}")
    
    try:
        # Use the enhanced processing with LLM reranking and validation for maximum accuracy
        final_answers, total_tokens = await process_query_accurate(payload)
        
        # Add the total token usage to a custom response header for evaluation
        response.headers["X-Token-Usage"] = str(total_tokens)
        
        # Return the correct HackRxResponse object with the answers
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        logger.error(f"An unexpected server error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected server error occurred."
        )