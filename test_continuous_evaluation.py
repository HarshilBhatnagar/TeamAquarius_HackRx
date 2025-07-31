#!/usr/bin/env python3
"""
Continuous Evaluation Testing Script
Simulates hackathon evaluation patterns with multiple test rounds and performance metrics.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, Any, List
from datetime import datetime

# Configuration
BASE_URL = "https://teamaquariushackrx-production-1bbc.up.railway.app"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test document
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Comprehensive test question sets (simulating hackathon evaluation)
TEST_SETS = {
    "Round 1 - Basic Policy Questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "What is the Moratorium Period defined in this policy?",
        "Does the policy cover maternity expenses?",
        "What is the No Claim Discount (NCD) offered?"
    ],
    
    "Round 2 - Coverage & Benefits": [
        "Are dental procedures covered under this policy?",
        "Does the policy cover pre-hospitalization medical expenses?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges?",
        "Is there a benefit for preventive health check-ups?"
    ],
    
    "Round 3 - Scenario-Based Questions": [
        "My hospital bill for ICU charges was Rs. 12,000 for one day. My Sum Insured is Rs. 5 Lakhs. How much will the policy pay?",
        "I need cataract surgery on both eyes, and the total bill is Rs. 90,000. My Sum Insured is Rs. 3 Lakhs. Will the full amount be paid?",
        "My total bill for a robotic surgery was Rs. 2,00,000. My Sum Insured is Rs. 3 Lakhs. How much will the policy pay for this treatment?",
        "I was hospitalized for plastic surgery to change my appearance. Is this covered?",
        "I was hospitalized for a routine dental check-up. Is this covered?"
    ],
    
    "Round 4 - Complex & Edge Cases": [
        "How does the policy define a 'Hospital'?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What happens if I miss premium payment by 45 days?",
        "Can I port this policy to another insurance company?",
        "What is the maximum Cumulative Bonus that can be accrued under the policy?"
    ],
    
    "Round 5 - Out-of-Domain & Trick Questions": [
        "Give me a recipe for chocolate cake.",
        "What is the ideal spark plug gap recommended?",
        "Please provide the Python code to connect to a PostgreSQL database.",
        "What is the capital of France?",
        "How do I make a paper airplane?"
    ]
}

async def test_single_round(round_name: str, questions: List[str], round_num: int) -> Dict[str, Any]:
    """Test a single round of questions."""
    print(f"\n🔄 {round_name}")
    print("=" * 80)
    
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
        
        # Analyze answers
        answers = result.get('answers', [])
        valid_answers = 0
        out_of_domain_detected = 0
        
        for i, answer in enumerate(answers):
            question = questions[i]
            answer_lower = answer.lower()
            
            # Check if out-of-domain was properly detected
            if any(phrase in answer_lower for phrase in [
                "not related to the insurance policy",
                "not related to the policy",
                "please ask questions about the policy"
            ]):
                out_of_domain_detected += 1
                print(f"   Q{i+1}: ✅ Out-of-domain properly detected")
            else:
                valid_answers += 1
                print(f"   Q{i+1}: ✅ Answered ({len(answer)} chars)")
        
        print(f"   ⏱️  Response time: {response_time:.2f}s")
        print(f"   📊 Token usage: {token_usage}")
        print(f"   📝 Valid answers: {valid_answers}/{len(questions)}")
        print(f"   🚫 Out-of-domain detected: {out_of_domain_detected}/{len(questions)}")
        
        return {
            "round_name": round_name,
            "round_num": round_num,
            "questions_count": len(questions),
            "response_time": response_time,
            "token_usage": token_usage,
            "valid_answers": valid_answers,
            "out_of_domain_detected": out_of_domain_detected,
            "success": True
        }
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            "round_name": round_name,
            "round_num": round_num,
            "questions_count": len(questions),
            "response_time": 0,
            "token_usage": "Error",
            "valid_answers": 0,
            "out_of_domain_detected": 0,
            "success": False
        }

async def test_concurrent_requests():
    """Test multiple concurrent requests to simulate load."""
    print(f"\n🚀 Testing Concurrent Requests")
    print("=" * 80)
    
    # Create 3 concurrent requests with different question sets
    concurrent_payloads = [
        {
            "documents": TEST_DOCUMENT,
            "questions": ["What is the grace period for premium payment?"]
        },
        {
            "documents": TEST_DOCUMENT,
            "questions": ["What is the waiting period for pre-existing diseases?"]
        },
        {
            "documents": TEST_DOCUMENT,
            "questions": ["Are dental procedures covered under this policy?"]
        }
    ]
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, payload in enumerate(concurrent_payloads):
                task = session.post(
                    f"{BASE_URL}/api/v1/hackrx/run",
                    headers=HEADERS,
                    json=payload
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            total_tokens = 0
            
            for i, response in enumerate(responses):
                result = await response.json()
                token_usage = response.headers.get("X-Token-Usage", "0")
                total_tokens += int(token_usage) if token_usage.isdigit() else 0
                print(f"   Request {i+1}: ✅ Success ({len(result.get('answers', []))} answers)")
            
            print(f"   ⏱️  Total concurrent time: {total_time:.2f}s")
            print(f"   📊 Total tokens: {total_tokens}")
            
            return {
                "concurrent_requests": len(concurrent_payloads),
                "total_time": total_time,
                "total_tokens": total_tokens,
                "success": True
            }
            
    except Exception as e:
        print(f"   ❌ Concurrent test error: {e}")
        return {
            "concurrent_requests": len(concurrent_payloads),
            "total_time": 0,
            "total_tokens": 0,
            "success": False
        }

async def main():
    """Main evaluation function."""
    print("🎯 Continuous Evaluation Testing Suite")
    print("=" * 80)
    print(f"🎯 Target: {BASE_URL}")
    print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    all_results = []
    
    # Test each round
    for round_num, (round_name, questions) in enumerate(TEST_SETS.items(), 1):
        result = await test_single_round(round_name, questions, round_num)
        all_results.append(result)
        
        # Small delay between rounds to simulate real evaluation
        await asyncio.sleep(2)
    
    # Test concurrent requests
    concurrent_result = await test_concurrent_requests()
    
    # Calculate comprehensive statistics
    successful_rounds = [r for r in all_results if r['success']]
    response_times = [r['response_time'] for r in successful_rounds]
    total_questions = sum(r['questions_count'] for r in successful_rounds)
    total_valid_answers = sum(r['valid_answers'] for r in successful_rounds)
    total_out_of_domain = sum(r['out_of_domain_detected'] for r in successful_rounds)
    
    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    print(f"🎯 Total Rounds: {len(all_results)}")
    print(f"✅ Successful Rounds: {len(successful_rounds)}")
    print(f"📝 Total Questions: {total_questions}")
    print(f"✅ Valid Answers: {total_valid_answers}")
    print(f"🚫 Out-of-Domain Detected: {total_out_of_domain}")
    
    if response_times:
        print(f"\n⏱️  RESPONSE TIME STATISTICS:")
        print(f"   Average: {statistics.mean(response_times):.2f}s")
        print(f"   Median: {statistics.median(response_times):.2f}s")
        print(f"   Min: {min(response_times):.2f}s")
        print(f"   Max: {max(response_times):.2f}s")
        print(f"   Standard Deviation: {statistics.stdev(response_times):.2f}s")
    
    # Performance assessment
    avg_response_time = statistics.mean(response_times) if response_times else 0
    success_rate = (len(successful_rounds) / len(all_results)) * 100
    accuracy_rate = (total_valid_answers / total_questions) * 100 if total_questions > 0 else 0
    
    print(f"\n🎯 PERFORMANCE ASSESSMENT:")
    print(f"   Response Time: {'✅ EXCELLENT' if avg_response_time < 30 else '⚠️ NEEDS IMPROVEMENT'} ({avg_response_time:.2f}s vs 30s target)")
    print(f"   Success Rate: {'✅ EXCELLENT' if success_rate >= 95 else '⚠️ NEEDS IMPROVEMENT'} ({success_rate:.1f}%)")
    print(f"   Accuracy Rate: {'✅ EXCELLENT' if accuracy_rate >= 75 else '⚠️ NEEDS IMPROVEMENT'} ({accuracy_rate:.1f}%)")
    print(f"   Out-of-Domain Detection: {'✅ EXCELLENT' if total_out_of_domain >= 4 else '⚠️ NEEDS IMPROVEMENT'} ({total_out_of_domain}/5 detected)")
    
    # Concurrent performance
    if concurrent_result['success']:
        print(f"   Concurrent Performance: ✅ EXCELLENT ({concurrent_result['total_time']:.2f}s for {concurrent_result['concurrent_requests']} requests)")
    
    print(f"\n⏰ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Final verdict
    if avg_response_time < 30 and success_rate >= 95 and accuracy_rate >= 75:
        print("🎉 EXCELLENT: Your API is ready for hackathon evaluation!")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Some metrics need optimization.")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(main()) 