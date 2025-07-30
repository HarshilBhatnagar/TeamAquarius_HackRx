import os
from openai import AsyncOpenAI
from utils.logger import logger
from typing import Tuple

try:
    validation_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

VALIDATION_PROMPT = """
You are an expert insurance policy validator with 15+ years of experience. Your task is to verify if the given answer is supported by the provided context for insurance-related questions.

**Validation Criteria:**
1. **Direct Support**: Answer must be directly derivable from the context
2. **Reasonable Inference**: Allow logical inferences from policy information (e.g., if policy covers "surgery" and question asks about "cataract surgery", it's valid)
3. **No Contradictions**: Answer should not contradict the context
4. **Factual Accuracy**: Numbers, percentages, and specific terms must match the context
5. **Logical Consistency**: Answer should follow logical reasoning

**Insurance Policy Validation Guidelines:**
- ✅ **Allow**: Reasonable inferences from general policy terms to specific procedures
- ✅ **Allow**: Calculations based on policy percentages and limits
- ✅ **Allow**: Answers that combine multiple policy clauses logically
- ✅ **Allow**: Inferences from general coverage terms to specific medical procedures
- ✅ **Allow**: Time-based calculations (waiting periods, grace periods)
- ❌ **Reject**: Answers that make assumptions not supported by the policy
- ❌ **Reject**: Contradictory information or incorrect calculations
- ❌ **Reject**: Information not present in the context

**Confidence Levels:**
- **High (0.9-1.0)**: Answer is clearly supported and accurate
- **Medium (0.6-0.8)**: Answer is mostly supported with minor uncertainties
- **Low (0.3-0.5)**: Answer has significant issues but some support
- **Very Low (0.1-0.2)**: Answer is poorly supported or incorrect

**Context:**
{context}

**Generated Answer:**
{answer}

**Question:**
{question}

**Validation Assessment:**
1. Is the answer supported by the context? (YES/NO)
2. Confidence level: (0.1-1.0)
3. Brief reasoning for your assessment:

**Response Format:**
```
SUPPORTED: YES/NO
CONFIDENCE: 0.X
REASONING: [Brief explanation]
```

If the answer is not well-supported, provide a corrected version:
CORRECTED_ANSWER: [Improved answer based on context]
"""

async def validate_answer(context: str, answer: str, question: str) -> Tuple[bool, str]:
    """
    Enhanced answer validation with confidence scoring and improved insurance-specific validation.
    Returns (is_valid, corrected_answer).
    """
    try:
        logger.info(f"Validating answer for question: '{question}'")

        validation_prompt = VALIDATION_PROMPT.format(
            context=context,
            answer=answer,
            question=question
        )

        response = await validation_client.chat.completions.create(
            messages=[{"role": "user", "content": validation_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000
        )

        validation_result = response.choices[0].message.content.strip()
        
        # Parse validation result
        is_supported, confidence, reasoning, corrected_answer = parse_validation_result(validation_result)
        
        logger.info(f"Validation result: Supported={is_supported}, Confidence={confidence}")
        logger.info(f"Reasoning: {reasoning}")

        # Determine if answer is valid based on confidence threshold
        confidence_float = float(confidence) if confidence else 0.5
        is_valid = is_supported and confidence_float >= 0.6

        if is_valid:
            return True, answer
        else:
            # Return corrected answer if available, otherwise return original
            final_answer = corrected_answer if corrected_answer else answer
            logger.warning(f"Answer validation failed. Using {'corrected' if corrected_answer else 'original'} answer")
            return False, final_answer

    except Exception as e:
        logger.error(f"Error in answer validation: {e}")
        # On error, assume answer is valid to avoid blocking responses
        return True, answer

def parse_validation_result(result: str) -> Tuple[bool, str, str, str]:
    """
    Parse the validation result from the LLM response.
    Returns (is_supported, confidence, reasoning, corrected_answer).
    """
    try:
        lines = result.split('\n')
        is_supported = False
        confidence = "0.5"
        reasoning = ""
        corrected_answer = ""

        for line in lines:
            line = line.strip()
            if line.startswith("SUPPORTED:"):
                is_supported = "YES" in line.upper()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":")[1].strip()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip() if ":" in line else ""
            elif line.startswith("CORRECTED_ANSWER:"):
                corrected_answer = line.split(":", 1)[1].strip() if ":" in line else ""

        return is_supported, confidence, reasoning, corrected_answer

    except Exception as e:
        logger.warning(f"Error parsing validation result: {e}")
        return True, "0.5", "Parsing error", "" 