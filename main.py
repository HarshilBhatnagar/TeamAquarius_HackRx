from fastapi import FastAPI
from routers import hackrx

app = FastAPI(
    title="HackRx API",
    description="API for processing documents and answering questions.",
    version="1.0.0",
    # Update docs URL to reflect the new prefix
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json"
)

# Include the router with the full required prefix
app.include_router(hackrx.router, prefix="/api/v1", tags=["HackRx"])

@app.get("/", tags=["Root"])
async def read_root():
    """
    A simple root endpoint to confirm the API is running.
    """
    return {"message": "Welcome to the HackRx API! Visit /api/v1/docs for more info."}