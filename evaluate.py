import asyncio
import json
import os
import time
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
API_ENDPOINT = "http://127.0.0.1:8000/api/v1/hackrx/run"
TEST_DATA_FILE = "evaluation/test_data.json"
API_TOKEN = os.getenv("API_AUTH_TOKEN")
EVALUATOR_CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EVALUATION_PROMPT_TEMPLATE = "You are an AI evaluator. Does the 'Predicted Answer' accurately match the 'Golden Answer'? Respond only with 'CORRECT' or 'INCORRECT'.\n\nGolden Answer: {golden_answer}\nPredicted Answer: {predicted_answer}"

async def get_api_response(document_url: str, question: str) -> dict:
    """Calls your API and returns the answer, latency, and token usage."""
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    payload = {"documents": document_url, "questions": [question]}
    
    start_time = time.time()
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(API_ENDPOINT, headers=headers, json=payload)
            latency = time.time() - start_time
            response.raise_for_status()
            
            return {
                "answer": response.json()["answers"][0],
                "latency": latency,
                "tokens": int(response.headers.get("x-token-usage", 0))
            }
        except Exception as e:
            return {"answer": f"Error: {e}", "latency": time.time() - start_time, "tokens": 0}

async def evaluate_answer(golden_answer: str, predicted_answer: str) -> bool:
    """Uses an LLM to evaluate if the predicted answer is correct."""
    prompt = EVALUATION_PROMPT_TEMPLATE.format(golden_answer=golden_answer, predicted_answer=predicted_answer)
    try:
        response = await EVALUATOR_CLIENT.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return "CORRECT" in response.choices[0].message.content.strip().upper()
    except Exception:
        return False

async def main():
    print("--- Starting Advanced Local Evaluation Suite ---")
    with open(TEST_DATA_FILE, "r") as f:
        test_cases = json.load(f)

    results = []
    for i, case in enumerate(test_cases):
        print(f"\nRunning Test Case {i+1}/{len(test_cases)}: {case['question']}")
        api_result = await get_api_response(case["document_url"], case["question"])
        is_correct = await evaluate_answer(case["golden_answer"], api_result["answer"])
        results.append({"correct": is_correct, **api_result})
        print(f"  Latency: {api_result['latency']:.2f}s | Tokens: {api_result['tokens']} | Correct: {'✅' if is_correct else '❌'}")

    # --- Calculate Final Predicted Score ---
    total_cases = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    accuracy = (correct_count / total_cases) * 100 if total_cases > 0 else 0
    avg_latency = sum(r["latency"] for r in results) / total_cases if total_cases > 0 else 0
    total_tokens = sum(r["tokens"] for r in results)

    # Simple scoring formula to predict leaderboard performance
    accuracy_score = accuracy
    latency_score = max(0, 30 - avg_latency) * 2  # Score based on how far under 30s you are
    token_score = max(0, 20000 - total_tokens) / 500 # Score based on token efficiency
    predicted_score = (0.6 * accuracy_score) + (0.3 * latency_score) + (0.1 * token_score)

    print("\n--- Evaluation Summary ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total_cases})")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Total Tokens Used: {total_tokens}")
    print(f"Predicted Score: {predicted_score:.2f}")
    print("--------------------------")

if __name__ == "__main__":
    asyncio.run(main())