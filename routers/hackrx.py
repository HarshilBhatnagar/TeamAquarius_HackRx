import json
from fastapi import APIRouter, Request, status, HTTPException, Depends
from fastapi.responses import StreamingResponse
from schemas.request import HackRxRequest
# The response model is no longer used directly in the endpoint signature
# from schemas.response import HackRxResponse 
from services.query_engine import process_query_stream # <-- Import the new generator function
from utils.logger import logger
from utils.security import validate_token

router = APIRouter()

# This async generator will construct the JSON stream
async def stream_generator(payload: HackRxRequest):
    yield '{"answers": ['
    is_first = True
    async for answer in process_query_stream(payload):
        if not is_first:
            yield ','
        # Escape the string and wrap it in quotes for valid JSON
        yield json.dumps(answer)
        is_first = False
    yield ']}'

@router.post(
    "/hackrx/run",
    # response_model is removed for streaming responses
    status_code=status.HTTP_200_OK,
    summary="Run the full RAG pipeline with streaming",
    dependencies=[Depends(validate_token)]
)
async def run_hackrx(payload: HackRxRequest, request: Request):
    ip_address = request.client.host
    logger.info(f"Authenticated streaming request from IP: {ip_address}")
    
    return StreamingResponse(
        stream_generator(payload),
        media_type="application/json"
    )