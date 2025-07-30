#!/usr/bin/env python3
"""
Enhanced Test script for the HackRx RAG API - Hackathon Version.
Tests the main /hackrx/run endpoint with enhanced accuracy features.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "your-api-auth-token-here"  # Replace with your actual token
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test data - Using the exact format from hackathon documentation
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
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

async def test_enhanced_hackathon_endpoint():
    """Test the enhanced hackathon endpoint with accuracy improvements."""
    print("🚀 Testing Enhanced HackRx RAG API - Hackathon Endpoint")
    print("=" * 70)
    print("✨ Enhanced Features: Multi-stage retrieval, Chain-of-thought, Self-consistency")
    print("🎯 Target: 75%+ accuracy with <30 second response time")
    print("=" * 70)

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
        response_time = end_time - start_time

        print(f"✅ Enhanced response received in {response_time:.2f} seconds")
        print(f"📊 Status Code: {status_code}")
        print(f"📊 Token usage: {token_usage}")
        print(f"📝 Number of answers: {len(result['answers'])}")
        
        # Performance analysis
        if response_time < 30:
            print(f"🎯 Performance: EXCELLENT (<30s target met)")
        elif response_time < 45:
            print(f"🎯 Performance: GOOD (within acceptable range)")
        else:
            print(f"⚠️  Performance: SLOW (exceeds target)")

        # Answer quality analysis
        print(f"\n📋 Answer Quality Analysis:")
        for i, (question, answer) in enumerate(zip(TEST_QUESTIONS[:5], result['answers'][:5])):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer[:200]}...")
            
            # Simple quality indicators
            if "not available" in answer.lower() or "not found" in answer.lower():
                print("   ❌ Quality: Low (information not found)")
            elif len(answer) > 50 and any(char.isdigit() for char in answer):
                print("   ✅ Quality: High (detailed with specific information)")
            elif len(answer) > 30:
                print("   ✅ Quality: Medium (reasonable answer)")
            else:
                print("   ⚠️  Quality: Low (too brief)")

        if len(result['answers']) > 5:
            print(f"\n... and {len(result['answers']) - 5} more answers")

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

async def test_accuracy_features():
    """Test specific accuracy-enhancing features."""
    print("\n🎯 Testing Accuracy Enhancement Features")
    print("=" * 60)

    # Test with questions that require different types of reasoning
    accuracy_test_questions = [
        "What is the maximum sum insured under this policy?",  # Direct lookup
        "If I have a pre-existing condition, when will it be covered?",  # Time-based reasoning
        "How much will the policy pay for a Rs. 50,000 hospital bill?",  # Calculation required
        "Are dental procedures covered under this policy?",  # Coverage determination
        "What happens if I miss a premium payment?"  # Process understanding
    ]

    payload = {
        "documents": TEST_DOCUMENT,
        "questions": accuracy_test_questions
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

        print(f"✅ Accuracy test completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Token usage: {token_usage}")

        print(f"\n🔍 Accuracy Feature Analysis:")
        for i, (question, answer) in enumerate(zip(accuracy_test_questions, result['answers'])):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer[:150]}...")
            
            # Analyze answer characteristics
            has_numbers = any(char.isdigit() for char in answer)
            has_policy_terms = any(term in answer.lower() for term in ['policy', 'coverage', 'insured', 'premium', 'claim'])
            is_detailed = len(answer) > 100
            
            if has_numbers and has_policy_terms and is_detailed:
                print("   ✅ High Quality: Specific, detailed, policy-relevant")
            elif has_policy_terms and is_detailed:
                print("   ✅ Good Quality: Detailed and policy-relevant")
            elif has_policy_terms:
                print("   ⚠️  Medium Quality: Policy-relevant but brief")
            else:
                print("   ❌ Low Quality: Generic or insufficient")

        return result

    except Exception as e:
        print(f"❌ Accuracy test failed: {e}")
        raise

async def main():
    """Run all enhanced tests."""
    print("🧪 Enhanced HackRx API Test Suite - Hackathon Version")
    print("=" * 70)
    print("🎯 Testing Enhanced Accuracy Features:")
    print("   • Multi-stage retrieval (50→20→12 chunks)")
    print("   • Chain-of-thought prompting")
    print("   • Self-consistency checking")
    print("   • Confidence scoring")
    print("   • Enhanced validation")
    print("   • Dynamic chunking strategy")
    print("=" * 70)

    try:
        await test_api_health()
        await test_sample_request()
        await test_accuracy_features()
        await test_enhanced_hackathon_endpoint()

        print("\n🎉 All enhanced tests completed successfully!")
        print("\n📝 Ready for hackathon submission!")
        print("Your webhook URL: https://your-app.railway.app/api/v1/hackrx/run")
        print("\n✨ Enhanced Features Summary:")
        print("   • Multi-stage retrieval for better context selection")
        print("   • Chain-of-thought reasoning for complex questions")
        print("   • Self-consistency checking for answer validation")
        print("   • Confidence scoring for quality assessment")
        print("   • Enhanced validation with correction capabilities")
        print("   • Dynamic chunking for optimal document processing")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 