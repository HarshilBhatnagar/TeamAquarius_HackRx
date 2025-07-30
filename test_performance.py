#!/usr/bin/env python3
"""
Performance Testing Script - Hackathon Evaluation
Tests response time optimizations to ensure <30 second average response time.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "your-api-auth-token-here"  # Replace with your actual token
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test document
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Performance test cases
PERFORMANCE_TEST_CASES = {
    "single_questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "What is the maximum sum insured under this policy?",
        "Are dental procedures covered under this policy?",
        "What is the co-payment applicable for an insured person aged 70?"
    ],
    
    "scenario_questions": [
        "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?",
        "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?",
        "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?",
        "My hospital bill for ICU charges was Rs 12,000 for one day. My Sum Insured is Rs 5 Lakhs. How much will the policy pay?",
        "I need cataract surgery on both eyes, and the total bill is Rs 90,000. My Sum Insured is Rs 3 Lakhs. Will the full amount be paid?"
    ],
    
    "out_of_domain": [
        "What is the ideal spark plug gap recommended?",
        "Give me JS code to generate a random number between 1 and 100",
        "What is the capital of France?",
        "How do I make a chocolate cake?",
        "Can you write a poem about insurance?"
    ],
    
    "mixed_batch": [
        "What is the grace period for premium payment?",
        "I have raised a claim for hospitalization for Rs 200,000. Can I raise the remaining Rs 50,000?",
        "What is the ideal spark plug gap recommended?",
        "What is the waiting period for pre-existing diseases?",
        "Give me JS code to connect to a database"
    ]
}

async def test_performance_batch(category: str, questions: List[str]) -> Dict[str, Any]:
    """Test performance for a batch of questions."""
    print(f"\n⚡ Testing {category.replace('_', ' ').title()} Performance")
    print("=" * 60)
    
    results = []
    total_time = 0
    total_tokens = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        
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
            total_time += response_time
            
            answer = result['answers'][0] if result['answers'] else "No answer received"
            
            # Extract token count
            if token_usage != "Unknown":
                try:
                    tokens = int(token_usage.split('/')[0])
                    total_tokens += tokens
                except:
                    pass
            
            print(f"   ⏱️  Response time: {response_time:.2f}s")
            print(f"   📊 Token usage: {token_usage}")
            print(f"   📝 Answer length: {len(answer)} chars")
            
            results.append({
                "question": question,
                "response_time": response_time,
                "token_usage": token_usage,
                "answer_length": len(answer)
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "question": question,
                "response_time": 0,
                "token_usage": "Error",
                "answer_length": 0
            })
    
    # Calculate statistics
    response_times = [r['response_time'] for r in results if r['response_time'] > 0]
    
    if response_times:
        avg_time = statistics.mean(response_times)
        median_time = statistics.median(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
    else:
        avg_time = median_time = min_time = max_time = std_dev = 0
    
    print(f"\n📊 {category.replace('_', ' ').title()} Performance Summary:")
    print(f"   Average Response Time: {avg_time:.2f}s")
    print(f"   Median Response Time: {median_time:.2f}s")
    print(f"   Min Response Time: {min_time:.2f}s")
    print(f"   Max Response Time: {max_time:.2f}s")
    print(f"   Standard Deviation: {std_dev:.2f}s")
    print(f"   Total Tokens Used: {total_tokens}")
    
    # Performance assessment
    if avg_time < 30:
        print(f"   🎯 Performance: EXCELLENT (<30s target met)")
    elif avg_time < 45:
        print(f"   🎯 Performance: GOOD (within acceptable range)")
    else:
        print(f"   ⚠️  Performance: SLOW (exceeds target)")
    
    return {
        "category": category,
        "results": results,
        "avg_time": avg_time,
        "median_time": median_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_dev": std_dev,
        "total_tokens": total_tokens,
        "success_count": len([r for r in results if r['response_time'] > 0])
    }

async def test_concurrent_performance():
    """Test concurrent processing performance."""
    print(f"\n🚀 Testing Concurrent Processing Performance")
    print("=" * 60)
    
    # Test with multiple questions in a single request
    questions = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Are dental procedures covered under this policy?",
        "What is the maximum sum insured under this policy?",
        "What is the co-payment applicable for an insured person aged 70?"
    ]
    
    payload = {
        "documents": TEST_DOCUMENT,
        "questions": questions
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
        
        print(f"📊 Concurrent Processing Results:")
        print(f"   Total Questions: {len(questions)}")
        print(f"   Total Response Time: {response_time:.2f}s")
        print(f"   Average per Question: {response_time/len(questions):.2f}s")
        print(f"   Token Usage: {token_usage}")
        
        # Performance assessment
        avg_per_question = response_time / len(questions)
        if avg_per_question < 30:
            print(f"   🎯 Performance: EXCELLENT (<30s per question)")
        elif avg_per_question < 45:
            print(f"   🎯 Performance: GOOD (within acceptable range)")
        else:
            print(f"   ⚠️  Performance: SLOW (exceeds target)")
        
        return {
            "total_questions": len(questions),
            "total_time": response_time,
            "avg_per_question": avg_per_question,
            "token_usage": token_usage
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

async def test_api_health():
    """Test basic API health."""
    print("🏥 Testing API Health")
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

async def main():
    """Run comprehensive performance tests."""
    print("🧪 Performance Testing Suite - Hackathon Evaluation")
    print("=" * 80)
    print("🎯 Testing Optimized Response Times:")
    print("   • Single Question Performance")
    print("   • Scenario-based Question Performance")
    print("   • Out-of-Domain Question Performance")
    print("   • Mixed Batch Performance")
    print("   • Concurrent Processing Performance")
    print("=" * 80)

    # Test API health first
    if not await test_api_health():
        print("❌ API health check failed. Exiting.")
        return

    # Test each category
    all_results = {}
    total_questions = 0
    total_time = 0
    
    for category, questions in PERFORMANCE_TEST_CASES.items():
        total_questions += len(questions)
        results = await test_performance_batch(category, questions)
        all_results[category] = results
        total_time += results['avg_time'] * len(questions)

    # Test concurrent processing
    concurrent_results = await test_concurrent_performance()

    # Overall summary
    print(f"\n🎉 Performance Test Results Summary")
    print("=" * 80)
    
    overall_avg_time = sum(r['avg_time'] for r in all_results.values()) / len(all_results)
    overall_success_rate = sum(r['success_count'] for r in all_results.values()) / total_questions
    
    print(f"📊 Overall Performance:")
    print(f"   Total Questions Tested: {total_questions}")
    print(f"   Overall Average Response Time: {overall_avg_time:.2f}s")
    print(f"   Overall Success Rate: {overall_success_rate:.1%}")
    
    print(f"\n📋 Category Breakdown:")
    for category, results in all_results.items():
        print(f"   {category.replace('_', ' ').title()}: {results['avg_time']:.2f}s avg, {results['success_count']}/{len(PERFORMANCE_TEST_CASES[category])} successful")
    
    if concurrent_results:
        print(f"\n🚀 Concurrent Processing:")
        print(f"   {concurrent_results['total_questions']} questions in {concurrent_results['total_time']:.2f}s")
        print(f"   Average per question: {concurrent_results['avg_per_question']:.2f}s")
    
    # Final performance assessment
    print(f"\n🎯 Final Performance Assessment:")
    if overall_avg_time < 30:
        print(f"   ✅ EXCELLENT: Average response time {overall_avg_time:.2f}s (<30s target met)")
    elif overall_avg_time < 45:
        print(f"   ✅ GOOD: Average response time {overall_avg_time:.2f}s (within acceptable range)")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Average response time {overall_avg_time:.2f}s (exceeds target)")
    
    if overall_success_rate >= 0.95:
        print(f"   ✅ EXCELLENT: Success rate {overall_success_rate:.1%} (≥95%)")
    elif overall_success_rate >= 0.90:
        print(f"   ✅ GOOD: Success rate {overall_success_rate:.1%} (≥90%)")
    else:
        print(f"   ⚠️  NEEDS IMPROVEMENT: Success rate {overall_success_rate:.1%} (<90%)")
    
    print(f"\n✨ Performance testing completed!")
    print(f"Your system is ready for hackathon evaluation!")

if __name__ == "__main__":
    asyncio.run(main()) 