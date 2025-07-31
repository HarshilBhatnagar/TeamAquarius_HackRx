#!/usr/bin/env python3
"""
Diagnostic test to understand why hackathon scores are low.
Tests various scenarios that evaluators might be using.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime

# API Configuration
API_BASE_URL = "https://teamaquariushackrx-production-1bbc.up.railway.app"
API_ENDPOINT = "/api/v1/hackrx/run"
AUTH_TOKEN = "8de3df5870b1015db720fe67f65ce68c4523c41b13a4c1d2fa00ce825d5f5a70"
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

# Different test scenarios that evaluators might be using
DIAGNOSTIC_TESTS = [
    {
        "name": "Basic Insurance Questions",
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "What is the Sum Insured under this policy?",
            "What is the premium amount?",
            "What is the policy term?",
            "What is the grace period?",
            "What is the waiting period?"
        ]
    },
    {
        "name": "Complex Scenario Questions",
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "I have diabetes and need hospitalization. Will it be covered?",
            "My wife is pregnant and needs delivery. Is maternity covered?",
            "I need a heart bypass surgery costing Rs. 10,00,000. My Sum Insured is Rs. 5 Lakhs. How much will be paid?",
            "Can I claim for dental treatment?",
            "What documents do I need for claim settlement?"
        ]
    },
    {
        "name": "Mixed Domain Questions",
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "What is the coverage for pre-existing diseases?",
            "How do I calculate compound interest?",
            "What is the claim settlement ratio?",
            "Can you write a Python function?",
            "What is the weather like today?"
        ]
    },
    {
        "name": "Edge Case Questions",
        "document": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "questions": [
            "What happens if I miss premium payment?",
            "Can I port this policy to another insurer?",
            "What is the moratorium period?",
            "Are alternative treatments like Ayurveda covered?",
            "What is the maximum number of claims I can make?"
        ]
    }
]

class DiagnosticTest:
    def __init__(self):
        self.results = []
        self.total_requests = 0
        self.successful_requests = 0
        self.response_times = []
        self.accuracy_scores = []
        
    async def test_single_case(self, session: aiohttp.ClientSession, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single document with its questions."""
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*60}")
        
        payload = {
            "documents": test_case["document"],
            "questions": test_case["questions"]
        }
        
        start_time = time.time()
        
        try:
            async with session.post(
                f"{API_BASE_URL}{API_ENDPOINT}",
                json=payload,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    result = await response.json()
                    self.successful_requests += 1
                    
                    print(f"✅ Success - Response Time: {response_time:.2f}s")
                    print(f"Questions: {len(test_case['questions'])}")
                    print(f"Answers: {len(result.get('answers', []))}")
                    
                    # Analyze answers for relevance and quality
                    relevance_score = self.analyze_answer_quality(test_case, result.get('answers', []))
                    
                    return {
                        "test_case": test_case["name"],
                        "status": "success",
                        "response_time": response_time,
                        "questions_count": len(test_case["questions"]),
                        "answers_count": len(result.get('answers', [])),
                        "relevance_score": relevance_score,
                        "answers": result.get('answers', [])
                    }
                else:
                    error_text = await response.text()
                    print(f"❌ Error {response.status}: {error_text}")
                    
                    return {
                        "test_case": test_case["name"],
                        "status": "error",
                        "response_time": response_time,
                        "error_code": response.status,
                        "error_message": error_text
                    }
                    
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ Exception: {str(e)}")
            
            return {
                "test_case": test_case["name"],
                "status": "exception",
                "response_time": response_time,
                "error_message": str(e)
            }
    
    def analyze_answer_quality(self, test_case: Dict[str, Any], answers: List[str]) -> float:
        """Analyze the quality and relevance of answers."""
        if not answers:
            return 0.0
        
        quality_score = 0
        total_questions = len(answers)
        
        for i, answer in enumerate(answers):
            answer_lower = answer.lower()
            question = test_case["questions"][i].lower()
            
            # Check for different quality indicators
            score = 0
            
            # 1. Check if answer is not empty
            if answer.strip():
                score += 0.2
            
            # 2. Check if answer is not generic
            generic_phrases = [
                "not available", "not provided", "not mentioned", "not specified",
                "information is not available", "data not available"
            ]
            if not any(phrase in answer_lower for phrase in generic_phrases):
                score += 0.3
            
            # 3. Check if answer contains relevant keywords
            insurance_keywords = ["policy", "coverage", "claim", "premium", "sum insured", "waiting period", "grace period", "hospitalization", "medical", "treatment", "disease", "surgery", "expenses", "documents", "settlement"]
            if any(keyword in answer_lower for keyword in insurance_keywords):
                score += 0.3
            
            # 4. Check if answer is not out-of-domain for insurance questions
            if "not related" in answer_lower or "out of domain" in answer_lower:
                # This is good for non-insurance questions
                if any(keyword in question for keyword in ["python", "weather", "calculate", "function", "code"]):
                    score += 0.2
                else:
                    score -= 0.2
            
            # 5. Check answer length (not too short, not too long)
            if 20 <= len(answer) <= 500:
                score += 0.2
            
            quality_score += min(score, 1.0)  # Cap at 1.0
        
        return quality_score / total_questions if total_questions > 0 else 0.0
    
    async def run_diagnostic_test(self):
        """Run the complete diagnostic test suite."""
        print("🔍 Starting Diagnostic Test")
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing {len(DIAGNOSTIC_TESTS)} scenarios with {sum(len(tc['questions']) for tc in DIAGNOSTIC_TESTS)} total questions")
        
        async with aiohttp.ClientSession() as session:
            for test_case in DIAGNOSTIC_TESTS:
                result = await self.test_single_case(session, test_case)
                self.results.append(result)
                self.total_requests += 1
                
                if result["status"] == "success":
                    self.response_times.append(result["response_time"])
                    self.accuracy_scores.append(result["relevance_score"])
                
                await asyncio.sleep(1)
        
        self.print_diagnostic_summary()
    
    def print_diagnostic_summary(self):
        """Print comprehensive diagnostic summary."""
        print(f"\n{'='*80}")
        print("🔍 DIAGNOSTIC TEST SUMMARY")
        print(f"{'='*80}")
        
        # Basic statistics
        print(f"📈 Total Test Scenarios: {len(DIAGNOSTIC_TESTS)}")
        print(f"🎯 Total Questions: {sum(len(tc['questions']) for tc in DIAGNOSTIC_TESTS)}")
        print(f"✅ Successful Requests: {self.successful_requests}/{self.total_requests}")
        print(f"📊 Success Rate: {(self.successful_requests/self.total_requests)*100:.1f}%")
        
        if self.response_times:
            print(f"\n⏱️  RESPONSE TIME STATISTICS:")
            print(f"   Average: {statistics.mean(self.response_times):.2f}s")
            print(f"   Median: {statistics.median(self.response_times):.2f}s")
            print(f"   Min: {min(self.response_times):.2f}s")
            print(f"   Max: {max(self.response_times):.2f}s")
        
        if self.accuracy_scores:
            print(f"\n🎯 QUALITY STATISTICS:")
            print(f"   Average Quality: {statistics.mean(self.accuracy_scores):.1%}")
            print(f"   Median Quality: {statistics.median(self.accuracy_scores):.1%}")
            print(f"   Min Quality: {min(self.accuracy_scores):.1%}")
            print(f"   Max Quality: {max(self.accuracy_scores):.1%}")
        
        # Detailed analysis
        print(f"\n📋 DETAILED ANALYSIS:")
        for result in self.results:
            status_emoji = "✅" if result["status"] == "success" else "❌"
            print(f"   {status_emoji} {result['test_case']}")
            print(f"      Status: {result['status']}")
            print(f"      Response Time: {result.get('response_time', 'N/A'):.2f}s")
            
            if result["status"] == "success":
                print(f"      Quality Score: {result.get('relevance_score', 0):.1%}")
                
                # Show sample answers
                if result.get('answers'):
                    print(f"      Sample Answers:")
                    for i, answer in enumerate(result['answers'][:3]):  # Show first 3
                        if len(answer) > 100:
                            answer = answer[:100] + "..."
                        print(f"        Q{i+1}: {answer}")
            else:
                print(f"      Error: {result.get('error_message', 'Unknown error')}")
            print()
        
        # Performance assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT:")
        if self.response_times:
            avg_time = statistics.mean(self.response_times)
            if avg_time <= 30:
                print(f"   ⚡ Speed: EXCELLENT (avg {avg_time:.1f}s ≤ 30s target)")
            elif avg_time <= 35:
                print(f"   🟡 Speed: GOOD (avg {avg_time:.1f}s, close to 30s target)")
            else:
                print(f"   🔴 Speed: NEEDS IMPROVEMENT (avg {avg_time:.1f}s > 30s target)")
        
        if self.accuracy_scores:
            avg_quality = statistics.mean(self.accuracy_scores)
            if avg_quality >= 0.75:
                print(f"   🎯 Quality: EXCELLENT ({avg_quality:.1%} ≥ 75% target)")
            elif avg_quality >= 0.60:
                print(f"   🟡 Quality: GOOD ({avg_quality:.1%}, approaching 75% target)")
            else:
                print(f"   🔴 Quality: NEEDS IMPROVEMENT ({avg_quality:.1%} < 75% target)")
        
        print(f"\n📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

async def main():
    """Main diagnostic execution."""
    tester = DiagnosticTest()
    await tester.run_diagnostic_test()

if __name__ == "__main__":
    asyncio.run(main()) 