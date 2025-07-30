#!/usr/bin/env python3
"""
Test Script for Deployed Railway Endpoint
Tests the live endpoint at https://teamaquariushackrx-production-1bbc.up.railway.app/
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, Any, List

# Configuration for deployed endpoint
BASE_URL = "https://teamaquariushackrx-production-1bbc.up.railway.app"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test document
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Quick test cases
QUICK_TESTS = [
    "What is the grace period for premium payment?",
    "What is the ideal spark plug gap recommended?",  # Out-of-domain test
    "What is the waiting period for pre-existing diseases?",
    "Give me JS code to connect to a database",  # Out-of-domain test
    "Are dental procedures covered under this policy?"
]

async def test_api_health():
    """Test basic API health."""
    print("🏥 Testing Deployed API Health")
    print("=" * 60)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/") as response:
                result = await response.json()

        print(f"✅ API is healthy: {result}")
        return True

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def test_single_question(question: str, question_num: int) -> Dict[str, Any]:
    """Test a single question."""
    print(f"\nQ{question_num}: {question}")
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": [question]
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
        response_time = end_time - start_time
        
        answer = result['answers'][0] if result['answers'] else "No answer received"
        
        print(f"   ⏱️  Response time: {response_time:.2f}s")
        print(f"   📊 Token usage: {token_usage}")
        print(f"   📝 Answer: {answer[:150]}...")
        
        return {
            "question": question,
            "response_time": response_time,
            "token_usage": token_usage,
            "answer": answer,
            "success": True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "question": question,
            "response_time": 0,
            "token_usage": "Error",
            "answer": f"Error: {e}",
            "success": False
        }

async def test_batch_questions():
    """Test multiple questions in a single request."""
    print(f"\n🚀 Testing Batch Processing")
    print("=" * 60)
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": QUICK_TESTS
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
        response_time = end_time - start_time
        
        print(f"📊 Batch Processing Results:")
        print(f"   Total Questions: {len(QUICK_TESTS)}")
        print(f"   Total Response Time: {response_time:.2f}s")
        print(f"   Average per Question: {response_time/len(QUICK_TESTS):.2f}s")
        print(f"   Token Usage: {token_usage}")
        
        answers = result.get('answers', [])
        for i, (question, answer) in enumerate(zip(QUICK_TESTS, answers)):
            print(f"   Q{i+1}: {answer[:100]}...")
        
        return {
            "total_questions": len(QUICK_TESTS),
            "total_time": response_time,
            "avg_per_question": response_time / len(QUICK_TESTS),
            "token_usage": token_usage,
            "success": True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "total_questions": len(QUICK_TESTS),
            "total_time": 0,
            "avg_per_question": 0,
            "token_usage": "Error",
            "success": False
        }

async def main():
    """Run comprehensive tests on deployed endpoint."""
    print("🧪 Deployed Endpoint Testing Suite")
    print("=" * 80)
    print(f"🎯 Testing: {BASE_URL}")
    print("=" * 80)

    # Test API health first
    if not await test_api_health():
        print("❌ API health check failed. Exiting.")
        return

    # Test individual questions
    print(f"\n📝 Testing Individual Questions")
    print("=" * 60)
    
    individual_results = []
    for i, question in enumerate(QUICK_TESTS, 1):
        result = await test_single_question(question, i)
        individual_results.append(result)
        await asyncio.sleep(1)  # Small delay between requests

    # Test batch processing
    batch_result = await test_batch_questions()

    # Calculate statistics
    successful_individual = [r for r in individual_results if r['success']]
    response_times = [r['response_time'] for r in successful_individual]
    
    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
    else:
        avg_time = median_time = min_time = max_time = 0

    # Overall summary
    print(f"\n🎉 Test Results Summary")
    print("=" * 80)
    
    print(f"📊 Individual Questions:")
    print(f"   Total Questions: {len(QUICK_TESTS)}")
    print(f"   Successful: {len(successful_individual)}")
    print(f"   Average Response Time: {avg_time:.2f}s")
    print(f"   Median Response Time: {median_time:.2f}s")
    print(f"   Min Response Time: {min_time:.2f}s")
    print(f"   Max Response Time: {max_time:.2f}s")
    
    if batch_result['success']:
        print(f"\n📊 Batch Processing:")
        print(f"   Total Questions: {batch_result['total_questions']}")
        print(f"   Total Response Time: {batch_result['total_time']:.2f}s")
        print(f"   Average per Question: {batch_result['avg_per_question']:.2f}s")
        print(f"   Token Usage: {batch_result['token_usage']}")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if avg_time < 30:
        print(f"   ✅ EXCELLENT: Average response time {avg_time:.2f}s (<30s target met)")
    elif avg_time < 45:
        print(f"   ✅ GOOD: Average response time {avg_time:.2f}s (within acceptable range)")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Average response time {avg_time:.2f}s (exceeds target)")
    
    success_rate = len(successful_individual) / len(QUICK_TESTS)
    if success_rate >= 0.95:
        print(f"   ✅ EXCELLENT: Success rate {success_rate:.1%} (≥95%)")
    elif success_rate >= 0.90:
        print(f"   ✅ GOOD: Success rate {success_rate:.1%} (≥90%)")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Success rate {success_rate:.1%} (<90%)")
    
    print(f"\n✨ Testing completed!")
    print(f"Your deployed endpoint is ready for hackathon evaluation!")

if __name__ == "__main__":
    asyncio.run(main()) 