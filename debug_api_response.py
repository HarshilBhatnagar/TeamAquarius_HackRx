#!/usr/bin/env python3
"""
Debug Script to Check API Response Format
"""

import asyncio
import aiohttp
import json

# Configuration
BASE_URL = "https://teamaquariushackrx-production-1bbc.up.railway.app"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

async def debug_api_response():
    """Debug the API response format."""
    print("🔍 Debugging API Response Format")
    print("=" * 60)
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": ["What is the grace period for premium payment?"]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=HEADERS,
                json=payload
            ) as response:
                print(f"Status Code: {response.status}")
                print(f"Headers: {dict(response.headers)}")
                
                result = await response.json()
                print(f"\nFull Response:")
                print(json.dumps(result, indent=2))
                
                print(f"\nResponse Keys: {list(result.keys())}")
                
                if 'answers' in result:
                    print(f"Answers: {result['answers']}")
                else:
                    print("No 'answers' key found in response")
                    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_api_response()) 