#!/usr/bin/env python3
"""
Test script for the HackRx RAG API - Hackathon Version.
Tests the main /hackrx/run endpoint with the exact format required by the hackathon.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test data - Using the exact format from hackathon documentation
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
TEST_QUESTIONS = [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?",
    "Does this policy cover maternity expenses, and what are the conditions?",
    "What is the waiting period for cataract surgery?",
    "Are the medical expenses for an organ donor covered under this policy?",
    "What is the No Claim Discount (NCD) offered in this policy?",
    "Is there a benefit for preventive health check-ups?",
    "How does the policy define a 'Hospital'?",
    "What is the extent of coverage for AYUSH treatments?",
    "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

async def test_hackathon_endpoint():
    """Test the main hackathon endpoint with the exact format required."""
    print("🚀 Testing HackRx RAG API - Hackathon Endpoint")
    print("=" * 60)
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": TEST_QUESTIONS
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=HEADERS,
                json=payload
            ) as response:
                result = await response.json()
                token_usage = response.headers.get("X-Token-Usage", "Unknown")
                status_code = response.status
        
        end_time = time.time()
        
        print(f"✅ Response received in {end_time - start_time:.2f} seconds")
        print(f"📊 Status Code: {status_code}")
        print(f"📊 Token usage: {token_usage}")
        print(f"📝 Number of answers: {len(result['answers'])}")
        
        # Display first few answers as examples
        for i, (question, answer) in enumerate(zip(TEST_QUESTIONS[:3], result['answers'][:3])):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer[:200]}...")
        
        if len(result['answers']) > 3:
            print(f"\n... and {len(result['answers']) - 3} more answers")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

async def test_api_health():
    """Test basic API health."""
    print("\n🏥 Testing API Health")
    print("=" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/") as response:
                result = await response.json()
        
        print(f"✅ API is healthy: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        raise

async def test_sample_request():
    """Test with the exact sample request from hackathon documentation."""
    print("\n📋 Testing Sample Request (from hackathon docs)")
    print("=" * 60)
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?"
        ]
    }
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}/api/v1/hackrx/run",
                headers=HEADERS,
                json=payload
            ) as response:
                result = await response.json()
                token_usage = response.headers.get("X-Token-Usage", "Unknown")
        
        end_time = time.time()
        
        print(f"✅ Sample request completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Token usage: {token_usage}")
        print(f"📝 Number of answers: {len(result['answers'])}")
        
        for i, (question, answer) in enumerate(zip(payload['questions'], result['answers'])):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer}")
        
        return result
        
    except Exception as e:
        print(f"❌ Sample request failed: {e}")
        raise

async def main():
    """Run all tests."""
    print("🧪 HackRx API Test Suite - Hackathon Version")
    print("=" * 60)
    
    try:
        # Test API health first
        await test_api_health()
        
        # Test sample request (smaller test)
        await test_sample_request()
        
        # Test full endpoint
        await test_hackathon_endpoint()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📝 Ready for hackathon submission!")
        print("Your webhook URL: https://your-app.railway.app/api/v1/hackrx/run")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 