import asyncio
import json
import os
import time
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# << --- ACTION REQUIRED: CHOOSE YOUR TEST TARGET --- >>
# Set to "local" to test your server running on http://127.0.0.1:8000
# Set to "deployed" to test your live Railway URL
TEST_ENVIRONMENT = "local" 

# --- Endpoints ---
API_ENDPOINT_LOCAL = "http://127.0.0.1:8000/api/v1/hackrx/run"
# IMPORTANT: Replace with your actual deployed URL from Railway
API_ENDPOINT_DEPLOYED = "https://your-app-name.up.railway.app/api/v1/hackrx/run" 

API_ENDPOINT = API_ENDPOINT_LOCAL if TEST_ENVIRONMENT == "local" else API_ENDPOINT_DEPLOYED
API_TOKEN = os.getenv("API_AUTH_TOKEN")
EVALUATOR_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add any new document URLs you unlock here to test them
TEST_DOCUMENTS = [
    "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D"
]

GENERATION_PROMPT = """
From the provided text from an insurance policy, generate 5 diverse question-and-answer pairs. The answers must be concise and directly derivable from the text.
Include a mix of the following question types:
1.  **Factual Questions**: Ask about specific details (e.g., "What is the waiting period for...?").
2.  **Scenario-Based Questions**: Create a short scenario (e.g., "I had a treatment that cost X, is it covered?").
3.  **Irrelevant/Trick Questions**: Ask one question that is completely unrelated to the document (e.g., "Give me JS code..." or "What is the capital of France?").

Respond with ONLY a valid JSON list of objects. Each object must have a "question" key and a "golden_answer" key.

Text:
{document_text}
"""
EVALUATION_PROMPT = "You are an AI evaluator. Does the 'Predicted Answer' accurately match the 'Golden Answer'? Respond only with 'CORRECT' or 'INCORRECT'.\n\nGolden Answer: {golden_answer}\nPredicted Answer: {predicted_answer}"

async def generate_test_cases(doc_url: str) -> list:
    print(f"--- Generating new test cases for document: {doc_url[:40]}... ---")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(doc_url, timeout=30.0)
            text_sample = response.text[:8000]

        prompt = GENERATION_PROMPT.format(document_text=text_sample)
        response = await EVALUATOR_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5
        )
        generated_json = json.loads(response.choices[0].message.content)
        for key, value in generated_json.items():
            if isinstance(value, list):
                print(f"  Successfully generated {len(value)} new test cases.")
                return value
        return []
    except Exception as e:
        print(f"  Error generating test cases: {e}")
        return []

async def get_api_response(doc_url: str, questions: list) -> dict:
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    payload = {"documents": doc_url, "questions": questions}
    
    start_time = time.time()
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            response = await client.post(API_ENDPOINT, headers=headers, json=payload)
            latency = time.time() - start_time
            response.raise_for_status()
            
            return {
                "answers": response.json()["answers"],
                "latency": latency,
                "tokens": int(response.headers.get("x-token-usage", 0))
            }
        except Exception as e:
            return {"answers": [f"Error: {e}" for _ in questions], "latency": time.time() - start_time, "tokens": 0}

async def evaluate_answers(test_cases: list, predicted_answers: list) -> list:
    results = []
    for i, case in enumerate(test_cases):
        is_correct = "INCORRECT"
        if i < len(predicted_answers):
            prompt = EVALUATION_PROMPT.format(golden_answer=case["golden_answer"], predicted_answer=predicted_answers[i])
            try:
                response = await EVALUATOR_CLIENT.chat.completions.create(
                    model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
                )
                if "CORRECT" in response.choices[0].message.content.strip().upper():
                    is_correct = "CORRECT"
            except Exception:
                pass
        results.append(is_correct)
    return results

async def main():
    print(f"--- Starting Dynamic Local Evaluation Suite ---")
    print(f"Targeting environment: {TEST_ENVIRONMENT.upper()} ({API_ENDPOINT})")
    all_results = []
    
    for doc_url in TEST_DOCUMENTS:
        test_cases = await generate_test_cases(doc_url)
        if not test_cases:
            continue

        questions = [case["question"] for case in test_cases]
        api_result = await get_api_response(doc_url, questions)
        
        evaluations = await evaluate_answers(test_cases, api_result["answers"])
        
        print("\n--- Evaluation Results ---")
        for i, case in enumerate(test_cases):
            result_icon = "✅" if evaluations[i] == "CORRECT" else "❌"
            print(f"  {result_icon} Q: {case['question']}")
            print(f"     Golden A: {case['golden_answer']}")
            print(f"   Predicted A: {api_result['answers'][i]}")

        correct_count = sum(1 for e in evaluations if e == "CORRECT")
        all_results.append({
            "correct": correct_count,
            "total": len(test_cases),
            "latency": api_result["latency"],
            "tokens": api_result["tokens"]
        })

    total_correct = sum(r["correct"] for r in all_results)
    total_questions = sum(r["total"] for r in all_results)
    accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    avg_latency = sum(r["latency"] for r in all_results) / len(all_results) if all_results else 0
    total_tokens = sum(r["tokens"] for r in all_results)

    accuracy_score = accuracy
    latency_score = max(0, 30 - avg_latency) * 2
    token_score = max(0, 50000 - total_tokens) / 1000
    predicted_score = (0.6 * accuracy_score) + (0.3 * latency_score) + (0.1 * token_score)

    print("\n--- Final Summary ---")
    print(f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"Average Latency per Document: {avg_latency:.2f}s")
    print(f"Total Tokens Used: {total_tokens}")
    print(f"Predicted Score: {predicted_score:.2f}")
    print("---------------------")

if __name__ == "__main__":
    asyncio.run(main())