import os
from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the security scheme
bearer_scheme = HTTPBearer()

# Get the valid token from environment variables
VALID_TOKEN = os.getenv("API_AUTH_TOKEN")
if not VALID_TOKEN:
    raise EnvironmentError("API_AUTH_TOKEN not found in .env file.")

async def validate_token(credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)):
    """
    Validates the bearer token from the Authorization header.
    """
    if not credentials or credentials.scheme != "Bearer" or credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials