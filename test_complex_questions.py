#!/usr/bin/env python3
"""
Comprehensive Test Script for Complex Question Types - Hackathon Evaluation
Tests scenario-based reasoning, quantitative lookups, exclusion identification, and out-of-domain queries.
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

# Test document
TEST_DOCUMENT = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"

# Complex question test cases
COMPLEX_QUESTIONS = {
    "scenario_based": [
        "My hospital bill for ICU charges was Rs. 12,000 for one day. My Sum Insured is Rs. 5 Lakhs. How much will the policy pay?",
        "I need cataract surgery on both eyes, and the total bill is Rs. 90,000. My Sum Insured is Rs. 3 Lakhs. Will the full amount be paid?",
        "I am 45 years old and have a pre-existing heart condition. What is my waiting period for coverage?",
        "My premium payment is due on 15th March, but I pay it on 20th March. Will my policy continue?",
        "I was hospitalized for 7 days with a covered illness. My room rent was Rs. 8,000 per day. What will be covered?"
    ],
    
    "quantitative_lookups": [
        "What is the maximum Cumulative Bonus that can be accrued under the policy?",
        "What is the sub-limit for cataract treatment?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "What is the maximum sum insured under this policy?",
        "What percentage of the sum insured is covered for room rent?",
        "What is the waiting period for maternity benefits?"
    ],
    
    "exclusion_identification": [
        "Are dental procedures covered under this policy?",
        "Does this policy cover cosmetic surgery?",
        "Are pre-existing conditions covered from day one?",
        "Is there coverage for alternative medicine treatments?",
        "Are routine health check-ups covered?",
        "Does the policy cover mental health conditions?",
        "Are there any exclusions for specific diseases?",
        "Is coverage available for organ transplant procedures?"
    ],
    
    "direct_policy_queries": [
        "What is the definition of a 'Hospital' under this policy?",
        "How does the policy define 'Pre-existing Disease'?",
        "What are the eligibility criteria for this policy?",
        "What documents are required for claim settlement?",
        "What is the policy renewal process?",
        "How is the sum insured calculated?",
        "What is the policy term?",
        "What are the premium payment modes available?"
    ],
    
    "out_of_domain": [
        "Please provide the Python code to connect to a PostgreSQL database.",
        "What is the capital of France?",
        "How do I make a chocolate cake?",
        "What is the current weather in New York?",
        "Can you write a poem about insurance?",
        "What is the meaning of life?",
        "How do I fix a flat tire?",
        "What are the latest news headlines?"
    ]
}

async def test_question_category(category: str, questions: List[str]) -> Dict[str, Any]:
    """Test a specific category of questions."""
    print(f"\n🎯 Testing {category.replace('_', ' ').title()} Questions")
    print("=" * 60)
    
    results = []
    total_time = 0
    
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
            
            # Analyze answer quality
            quality_score = analyze_answer_quality(question, answer, category)
            
            print(f"   ⏱️  Response time: {response_time:.2f}s")
            print(f"   📊 Token usage: {token_usage}")
            print(f"   📝 Answer: {answer[:150]}...")
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

def analyze_answer_quality(question: str, answer: str, category: str) -> int:
    """Analyze the quality of an answer based on the question category."""
    try:
        score = 0
        
        # Check for out-of-domain responses
        if category == "out_of_domain":
            if "not related" in answer.lower() or "not available" in answer.lower():
                return 10  # Perfect response for out-of-domain
            elif len(answer) < 50:
                return 5   # Brief response
            else:
                return 2   # Should have rejected out-of-domain
        
        # Check for empty or error responses
        if not answer or "error" in answer.lower() or "not available" in answer.lower():
            return 0
        
        # Length-based scoring
        if len(answer) > 100:
            score += 3
        elif len(answer) > 50:
            score += 2
        else:
            score += 1
        
        # Content-based scoring
        if any(char.isdigit() for char in answer):
            score += 2  # Contains specific information
        
        if any(term in answer.lower() for term in ['policy', 'coverage', 'insured', 'premium', 'claim']):
            score += 2  # Contains insurance terms
        
        # Category-specific scoring
        if category == "scenario_based":
            if any(term in answer.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%']):
                score += 2  # Contains calculations
            if any(term in answer.lower() for term in ['will', 'would', 'based on', 'according to']):
                score += 1  # Shows reasoning
        
        elif category == "quantitative_lookups":
            if any(char.isdigit() for char in answer):
                score += 3  # Contains specific numbers
            if any(term in answer.lower() for term in ['rs.', 'rupees', 'lakh', 'thousand', 'percent', '%', 'days', 'months', 'years']):
                score += 2  # Contains specific amounts/timeframes
        
        elif category == "exclusion_identification":
            if any(term in answer.lower() for term in ['covered', 'not covered', 'excluded', 'included']):
                score += 3  # Clear coverage determination
            if any(term in answer.lower() for term in ['yes', 'no', 'covered', 'excluded']):
                score += 1  # Clear yes/no response
        
        elif category == "direct_policy_queries":
            if len(answer) > 80:
                score += 2  # Detailed explanation
            if any(term in answer.lower() for term in ['defined', 'means', 'refers to', 'includes']):
                score += 1  # Shows definition/explanation
        
        return min(score, 10)  # Cap at 10
        
    except Exception as e:
        print(f"Error in quality analysis: {e}")
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
    """Run comprehensive tests for all question types."""
    print("🧪 Comprehensive Complex Question Test Suite - Hackathon Evaluation")
    print("=" * 80)
    print("🎯 Testing Enhanced Features for Complex Question Types:")
    print("   • Scenario-based Reasoning (applying policy rules to specific situations)")
    print("   • Quantitative Lookups (finding specific amounts, limits, timeframes)")
    print("   • Exclusion Identification (determining coverage vs exclusions)")
    print("   • Direct Policy Queries (simple lookups)")
    print("   • Out-of-Domain Queries (with strong guardrails)")
    print("=" * 80)

    # Test API health first
    if not await test_api_health():
        print("❌ API health check failed. Exiting.")
        return

    # Test each category
    all_results = {}
    total_questions = 0
    total_time = 0
    
    for category, questions in COMPLEX_QUESTIONS.items():
        total_questions += len(questions)
        results = await test_question_category(category, questions)
        all_results[category] = results
        total_time += results['avg_time'] * len(questions)

    # Overall summary
    print(f"\n🎉 Comprehensive Test Results Summary")
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