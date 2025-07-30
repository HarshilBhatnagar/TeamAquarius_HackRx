#!/usr/bin/env python3
"""
Test Script Based on Deployment Logs Analysis - Hackathon Evaluation
Mirrors the exact testing patterns observed in the deployment logs.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List

# Configuration
BASE_URL = "http://localhost:8000"
AUTH_TOKEN = "your-api-auth-token-here"  # Replace with your actual token
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Test documents from deployment logs
TEST_DOCUMENTS = {
    "insurance_policy": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "vehicle_manual": "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
    "constitution": "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"
}

# Test questions based on deployment logs analysis
DEPLOYMENT_TEST_CASES = {
    "out_of_domain_automotive": [
        "What is the ideal spark plug gap recommended",
        "Is it compulsoury to have a disc brake",
        "Does this comes in tubeless tyre version",
        "Can I put thums up instead of oil"
    ],
    
    "out_of_domain_programming": [
        "Give me JS code to generate a random number between 1 and 100",
        "How do I connect to a PostgreSQL database?",
        "What is the syntax for Python functions?",
        "How do I create an API endpoint?"
    ],
    
    "out_of_domain_general": [
        "What is the capital of France?",
        "How do I make a chocolate cake?",
        "What is the current weather in New York?",
        "What is the meaning of life?"
    ],
    
    "insurance_scenario_based": [
        "I have raised a claim for hospitalization for Rs 200,000 with HDFC, and it's approved. My total expenses are Rs 250,000. Can I raise the remaining Rs 50,000 with you?",
        "I renewed my policy yesterday, and I have been a customer for the last 6 years. Can I raise a claim for Hydrocele?",
        "When will my root canal claim of Rs 25,000 be settled?",
        "I have done an IVF for Rs 56,000. Is it covered?",
        "I did a cataract treatment of Rs 100,000. Will you settle the full Rs 100,000?"
    ],
    
    "insurance_coverage_questions": [
        "Is Non-infective Arthritis covered?",
        "Is abortion covered?",
        "Are road ambulance expenses covered, and what is the limit?",
        "What is the sub-limit for cataract treatment?",
        "What is the daily limit for Room Rent and ICU charges?"
    ],
    
    "insurance_policy_details": [
        "What is the grace period for premium payment in the Arogya Sanjeevani Policy?",
        "What is the waiting period for pre-existing diseases under this policy?",
        "What is the co-payment applicable for an insured person aged 70?",
        "What is the maximum cumulative bonus (CB) that can be accrued?",
        "How long is the waiting period for joint replacement surgery, unless it arises from an accident?"
    ],
    
    "insurance_documentation": [
        "Give me a list of documents to be uploaded for hospitalization for heart surgery.",
        "What documents are required for claim settlement?",
        "How are 'Modern Treatments' like Robotic Surgery covered?",
        "Are AYUSH treatments covered under this policy?"
    ]
}

async def test_deployment_pattern(category: str, questions: List[str], document_url: str) -> Dict[str, Any]:
    """Test a specific deployment pattern category."""
    print(f"\n🎯 Testing {category.replace('_', ' ').title()}")
    print("=" * 60)
    
    results = []
    total_time = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}: {question}")
        
        payload = {
            "documents": document_url,
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
            
            # Analyze response quality
            quality_score = analyze_deployment_response(question, answer, category)
            
            print(f"   ⏱️  Response time: {response_time:.2f}s")
            print(f"   📊 Token usage: {token_usage}")
            print(f"   📝 Answer: {answer[:200]}...")
            print(f"   🎯 Quality Score: {quality_score}/10")
            
            results.append({
                "question": question,
                "answer": answer,
                "response_time": response_time,
                "token_usage": token_usage,
                "quality_score": quality_score
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"Error: {e}",
                "response_time": 0,
                "token_usage": "Unknown",
                "quality_score": 0
            })
    
    # Calculate category statistics
    avg_time = total_time / len(questions) if questions else 0
    avg_quality = sum(r['quality_score'] for r in results) / len(results) if results else 0
    success_rate = len([r for r in results if r['quality_score'] > 0]) / len(results) if results else 0
    
    print(f"\n📊 {category.replace('_', ' ').title()} Summary:")
    print(f"   Average Response Time: {avg_time:.2f}s")
    print(f"   Average Quality Score: {avg_quality:.1f}/10")
    print(f"   Success Rate: {success_rate:.1%}")
    
    return {
        "category": category,
        "results": results,
        "avg_time": avg_time,
        "avg_quality": avg_quality,
        "success_rate": success_rate
    }

def analyze_deployment_response(question: str, answer: str, category: str) -> int:
    """Analyze response quality based on deployment logs patterns."""
    try:
        score = 0
        
        # Out-of-domain questions should be rejected
        if category.startswith("out_of_domain"):
            if "not related" in answer.lower() or "not available" in answer.lower():
                return 10  # Perfect rejection
            elif len(answer) < 100 and any(term in answer.lower() for term in ['sorry', 'cannot', 'unable']):
                return 8   # Good rejection
            else:
                return 2   # Should have rejected but didn't
        
        # Insurance questions should be answered appropriately
        if category.startswith("insurance"):
            # Check for empty or error responses
            if not answer or "error" in answer.lower():
                return 0
            
            # Length-based scoring
            if len(answer) > 150:
                score += 3
            elif len(answer) > 100:
                score += 2
            else:
                score += 1
            
            # Content-based scoring
            if any(char.isdigit() for char in answer):
                score += 2  # Contains specific information
            
            if any(term in answer.lower() for term in ['policy', 'coverage', 'insured', 'premium', 'claim']):
                score += 2  # Contains insurance terms
            
            # Category-specific scoring
            if category == "insurance_scenario_based":
                if any(term in answer.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%']):
                    score += 2  # Contains calculations
                if any(term in answer.lower() for term in ['will', 'would', 'based on', 'according to']):
                    score += 1  # Shows reasoning
            
            elif category == "insurance_coverage_questions":
                if any(term in answer.lower() for term in ['covered', 'not covered', 'excluded', 'included']):
                    score += 3  # Clear coverage determination
                if any(term in answer.lower() for term in ['yes', 'no', 'covered', 'excluded']):
                    score += 1  # Clear yes/no response
            
            elif category == "insurance_policy_details":
                if any(char.isdigit() for char in answer):
                    score += 3  # Contains specific numbers
                if any(term in answer.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%', 'days', 'months', 'years']):
                    score += 2  # Contains specific amounts/timeframes
            
            elif category == "insurance_documentation":
                if len(answer) > 120:
                    score += 2  # Detailed explanation
                if any(term in answer.lower() for term in ['document', 'required', 'upload', 'submit']):
                    score += 2  # Mentions documentation
        
        return min(score, 10)  # Cap at 10
        
    except Exception as e:
        print(f"Error in deployment response analysis: {e}")
        return 5  # Default score

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
    """Run deployment pattern tests."""
    print("🧪 Deployment Pattern Test Suite - Based on Actual Logs")
    print("=" * 80)
    print("🎯 Testing Patterns Observed in Deployment Logs:")
    print("   • Out-of-Domain Questions (automotive, programming, general)")
    print("   • Insurance Scenario-Based Reasoning")
    print("   • Insurance Coverage Questions")
    print("   • Insurance Policy Details")
    print("   • Insurance Documentation Requirements")
    print("=" * 80)

    # Test API health first
    if not await test_api_health():
        print("❌ API health check failed. Exiting.")
        return

    # Test each category
    all_results = {}
    total_questions = 0
    total_time = 0
    
    # Test out-of-domain questions with insurance document (should be rejected)
    for category, questions in DEPLOYMENT_TEST_CASES.items():
        if category.startswith("out_of_domain"):
            total_questions += len(questions)
            results = await test_deployment_pattern(category, questions, TEST_DOCUMENTS["insurance_policy"])
            all_results[category] = results
            total_time += results['avg_time'] * len(questions)
    
    # Test insurance questions with insurance document
    for category, questions in DEPLOYMENT_TEST_CASES.items():
        if category.startswith("insurance"):
            total_questions += len(questions)
            results = await test_deployment_pattern(category, questions, TEST_DOCUMENTS["insurance_policy"])
            all_results[category] = results
            total_time += results['avg_time'] * len(questions)

    # Overall summary
    print(f"\n🎉 Deployment Pattern Test Results Summary")
    print("=" * 80)
    
    overall_avg_quality = sum(r['avg_quality'] for r in all_results.values()) / len(all_results)
    overall_success_rate = sum(r['success_rate'] for r in all_results.values()) / len(all_results)
    overall_avg_time = total_time / total_questions if total_questions > 0 else 0
    
    print(f"📊 Overall Performance:")
    print(f"   Total Questions Tested: {total_questions}")
    print(f"   Average Response Time: {overall_avg_time:.2f}s")
    print(f"   Overall Quality Score: {overall_avg_quality:.1f}/10")
    print(f"   Overall Success Rate: {overall_success_rate:.1%}")
    
    print(f"\n📋 Category Breakdown:")
    for category, results in all_results.items():
        print(f"   {category.replace('_', ' ').title()}: {results['avg_quality']:.1f}/10 quality, {results['success_rate']:.1%} success")
    
    # Performance assessment
    if overall_avg_time < 30:
        print(f"🎯 Performance: EXCELLENT (<30s target met)")
    elif overall_avg_time < 45:
        print(f"🎯 Performance: GOOD (within acceptable range)")
    else:
        print(f"⚠️  Performance: SLOW (exceeds target)")
    
    if overall_avg_quality >= 7.5:
        print(f"🎯 Quality: EXCELLENT (≥7.5/10)")
    elif overall_avg_quality >= 6.0:
        print(f"🎯 Quality: GOOD (≥6.0/10)")
    else:
        print(f"⚠️  Quality: NEEDS IMPROVEMENT (<6.0/10)")
    
    print(f"\n✨ Ready for hackathon evaluation!")
    print(f"Your webhook URL: https://your-app.railway.app/api/v1/hackrx/run")

if __name__ == "__main__":
    asyncio.run(main()) 