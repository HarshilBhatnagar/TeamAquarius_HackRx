#!/usr/bin/env python3
"""
Check Deployment Status and Error Details
"""

import asyncio
import aiohttp
import json

# Configuration
BASE_URL = "https://teamaquariushackrx-production-1bbc.up.railway.app"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"

async def check_deployment():
    """Check deployment status and get error details."""
    print("🔍 Checking Deployment Status")
    print("=" * 60)
    
    # Test 1: Health endpoint (no auth required)
    print("\n1. Testing Health Endpoint:")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/") as response:
                print(f"   Status: {response.status}")
                result = await response.json()
                print(f"   Response: {result}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Docs endpoint
    print("\n2. Testing Docs Endpoint:")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/docs") as response:
                print(f"   Status: {response.status}")
                print(f"   Content-Type: {response.headers.get('content-type', 'Unknown')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: API endpoint with minimal payload
    print("\n3. Testing API Endpoint with Minimal Payload:")
    try:
        payload = {
            "documents": "https://example.com/test.pdf",
            "questions": ["test question"]
        }
        
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=headers,
                json=payload
            ) as response:
                print(f"   Status: {response.status}")
                print(f"   Headers: {dict(response.headers)}")
                
                if response.status == 500:
                    print("   ⚠️  Server Error - Checking for detailed error...")
                    try:
                        error_text = await response.text()
                        print(f"   Error Response: {error_text}")
                    except:
                        print("   Could not read error response")
                else:
                    result = await response.json()
                    print(f"   Response: {json.dumps(result, indent=2)}")
                    
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 4: Check if it's a memory issue
    print("\n4. Testing with Different Document URL:")
    try:
        payload = {
            "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
            "questions": ["What is this document about?"]
        }
        
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 500:
                    error_text = await response.text()
                    print(f"   Error: {error_text}")
                else:
                    result = await response.json()
                    print(f"   Response: {json.dumps(result, indent=2)}")
                    
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_deployment()) 