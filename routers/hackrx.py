from fastapi import APIRouter, Request, Response, status, HTTPException, Depends
from schemas.request import HackRxRequest
from schemas.response import HackRxResponse 
from services.query_engine import process_query
from utils.logger import logger
from utils.security import validate_token

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=HackRxResponse, 
    status_code=status.HTTP_200_OK,
    summary="Run the full RAG pipeline",
    dependencies=[Depends(validate_token)]
)
async def run_hackrx(payload: HackRxRequest, request: Request, response: Response):
    ip_address = request.client.host
    logger.info(f"Authenticated request from IP: {ip_address}")
    
    try:
        # 1. Await the final list of answers and the token count
        final_answers, total_tokens = await process_query(payload)
        
        # 2. Add the total token usage to a custom response header
        response.headers["X-Token-Usage"] = str(total_tokens)
        
        # 3. Return the correct HackRxResponse object with the answers
        return HackRxResponse(answers=final_answers)

    except Exception as e:
        logger.error(f"An unexpected server error occurred: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected server error occurred."
        )