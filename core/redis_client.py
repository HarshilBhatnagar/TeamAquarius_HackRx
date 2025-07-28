import os
import redis
from dotenv import load_dotenv

load_dotenv()

try:
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        raise ValueError("REDIS_URL not found in environment variables.")
    # use decode_responses=False to handle pickled objects (bytes)
    redis_client = redis.from_url(redis_url, decode_responses=False)
    redis_client.ping() # Check connection
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"Could not connect to Redis: {e}")
    redis_client = None