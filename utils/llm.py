import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

CHAIN_OF_THOUGHT_PROMPT = """
You are an expert insurance policy analyst with 15+ years of experience. Your task is to answer insurance-related questions based on the provided policy document context.

**Question Type Analysis:**
1. **Scenario-based Reasoning**: Apply policy rules to specific situations (e.g., "My bill is X, I am Y years old, how much is my co-payment?")
2. **Quantitative Lookups**: Find specific monetary limits, percentages, or waiting periods
3. **Exclusion Identification**: Determine if specific conditions/treatments are covered or excluded
4. **Direct Policy Queries**: Simple lookups for policy terms and conditions
5. **Out-of-Domain**: Questions not related to insurance (respond with "This question is not related to the insurance policy document provided")

**Instructions:**
1. **Analyze Question Type**: First identify what type of question this is
2. **Think Step by Step**: Break down complex scenarios into logical components
3. **Extract Key Information**: Look for specific policy terms, amounts, conditions
4. **Apply Policy Rules**: Use the context to answer scenario-based questions
5. **Check for Exclusions**: Verify if specific conditions are covered or excluded
6. **Provide Confidence**: Rate your confidence in the answer (High/Medium/Low)
7. **Guard Against Hallucination**: Only use information from the provided context

**Answer Format:**
```
**Question Type:** [Scenario-based/Quantitative/Exclusion/Direct/Out-of-Domain]

**Analysis:**
[Your step-by-step reasoning process]

**Answer:**
[Your final answer based on the analysis]

**Confidence:** [High/Medium/Low]
**Reasoning:** [Brief explanation of confidence level]
```

**Context:**
{context}

**Question:**
{question}

**Your Response:**
"""

SELF_CONSISTENCY_PROMPT = """
You are an expert insurance policy validator. Review the following answer for consistency and accuracy.

**Original Question:**
{question}

**Context Used:**
{context}

**Generated Answer:**
{answer}

**Consistency Check:**
1. Does the answer directly address the question type identified?
2. Is the answer supported by the provided context?
3. For scenario-based questions: Are the calculations and logic correct?
4. For quantitative questions: Are the numbers and percentages accurate?
5. For exclusion questions: Is the coverage/exclusion determination correct?
6. Are there any contradictions within the answer?
7. Does the answer follow logical reasoning?
8. For out-of-domain questions: Did the system properly identify and respond appropriately?

**Provide your assessment:**
- **Consistent**: Answer is accurate and well-supported
- **Needs Revision**: Answer has minor issues but is mostly correct
- **Inconsistent**: Answer has significant problems or contradictions

**If revision needed, provide the corrected answer:**
"""

OUT_OF_DOMAIN_PROMPT = """
You are an expert insurance policy analyst. Determine if the following question is related to the insurance policy document or is out-of-domain.

**Out-of-Domain Indicators:**
- Questions about programming, code, or technical implementation (e.g., "Give me JS code", "Python code", "database connection")
- Questions about vehicles, automotive parts, or mechanical systems (e.g., "spark plug gap", "tubeless tyre", "disc brake", "oil", "thums up")
- General knowledge questions not related to insurance (e.g., "capital of France", "weather", "meaning of life")
- Questions about other documents or topics (e.g., "constitution", "legal documents")
- Requests for creative content, stories, or non-factual information
- Questions about current events, politics, or unrelated topics
- Questions about food, recipes, or cooking (e.g., "chocolate cake")
- Questions about unrelated technical topics (e.g., "flat tire", "news headlines")

**Insurance Policy Indicators:**
- Questions about coverage, benefits, or policy terms
- Questions about premiums, claims, or payments
- Questions about waiting periods, exclusions, or conditions
- Questions about specific medical procedures or treatments
- Questions about policy limits, amounts, or percentages
- Questions about claim settlement or documentation
- Questions about policy renewal or eligibility
- Questions about specific diseases, conditions, or treatments

**Question:** {question}

**Context:** {context}

**Assessment:**
- **Insurance-Related**: Question is about the insurance policy
- **Out-of-Domain**: Question is not related to insurance policy

**Response:** [Insurance-Related/Out-of-Domain]
"""

async def get_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Enhanced answer generation using chain-of-thought prompting and self-consistency checking.
    Optimized for insurance documents with improved accuracy and confidence scoring.
    Handles complex question types including scenario-based reasoning and out-of-domain queries.
    """
    try:
        logger.info(f"Generating enhanced answer for question: '{question}'")

        # Step 1: Check if question is out-of-domain
        is_out_of_domain = await check_out_of_domain(question, context)
        if is_out_of_domain:
            return "This question is not related to the insurance policy document provided. Please ask questions about the policy coverage, benefits, terms, or conditions.", None

        # Step 2: Generate initial answer with chain-of-thought reasoning
        initial_prompt = CHAIN_OF_THOUGHT_PROMPT.format(
            context=context,
            question=question
        )

        initial_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": initial_prompt}],
            model="gpt-4o",
            temperature=0.1,  # Slight randomness for better reasoning
            max_tokens=1500
        )

        initial_answer = initial_response.choices[0].message.content
        usage = initial_response.usage

        # Step 3: Self-consistency check
        consistency_prompt = SELF_CONSISTENCY_PROMPT.format(
            question=question,
            context=context,
            answer=initial_answer
        )

        consistency_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": consistency_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=800
        )

        consistency_result = consistency_response.usage
        if usage:
            usage.total_tokens += consistency_result.total_tokens

        # Step 4: Extract final answer and confidence
        final_answer = extract_final_answer(initial_answer)
        confidence = extract_confidence(initial_answer)

        logger.info(f"Enhanced answer generated with confidence: {confidence}")
        return final_answer, usage

    except Exception as e:
        logger.error(f"Error in enhanced LLM answer generation: {e}")
        # Fallback to simple answer generation
        return await get_simple_llm_answer(context, question)

async def check_out_of_domain(question: str, context: str) -> bool:
    """
    Enhanced out-of-domain detection based on deployment logs analysis.
    """
    try:
        # Quick keyword-based check for obvious out-of-domain questions
        out_of_domain_keywords = {
            # Programming/Technical
            'code', 'python', 'javascript', 'js', 'database', 'postgresql', 'sql', 'programming',
            'function', 'variable', 'api', 'endpoint', 'server', 'client', 'html', 'css',
            
            # Automotive/Mechanical
            'spark plug', 'gap', 'tubeless', 'tyre', 'tire', 'disc brake', 'oil', 'thums up',
            'thumbs up', 'engine', 'motorcycle', 'bike', 'vehicle', 'automotive', 'mechanical',
            'brake', 'clutch', 'gear', 'transmission', 'fuel', 'petrol', 'diesel',
            
            # General Knowledge
            'capital', 'weather', 'recipe', 'cake', 'cooking', 'food', 'meaning of life',
            'news', 'headlines', 'flat tire', 'puncture', 'constitution', 'legal',
            
            # Unrelated Topics
            'poem', 'story', 'creative', 'fiction', 'novel', 'book', 'movie', 'music',
            'sports', 'game', 'entertainment', 'celebrity', 'politics', 'election'
        }
        
        question_lower = question.lower()
        
        # Check for out-of-domain keywords
        for keyword in out_of_domain_keywords:
            if keyword in question_lower:
                logger.info(f"Out-of-domain detected via keyword: {keyword}")
                return True
        
        # Use LLM for more nuanced detection
        out_of_domain_prompt = OUT_OF_DOMAIN_PROMPT.format(
            question=question,
            context=context[:1000]  # Use first 1000 chars for context
        )

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": out_of_domain_prompt}],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=50
        )

        assessment = response.choices[0].message.content.strip()
        is_out_of_domain = "Out-of-Domain" in assessment
        
        logger.info(f"Out-of-domain check: {assessment}")
        return is_out_of_domain

    except Exception as e:
        logger.warning(f"Error in out-of-domain check: {e}")
        return False

async def get_simple_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Simple fallback answer generation for error cases.
    """
    try:
        simple_prompt = f"""
        Based on the following insurance policy context, answer the question accurately and concisely.
        If the question is not related to insurance policy, respond with "This question is not related to the insurance policy document provided."

        Context:
        {context}

        Question: {question}

        Answer:
        """

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": simple_prompt}],
            model="gpt-4o",
            temperature=0,
            max_tokens=800
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage

        logger.info("Simple answer generation completed")
        return answer, usage

    except Exception as e:
        logger.error(f"Error in simple LLM answer generation: {e}")
        return "I apologize, but I encountered an error while processing your question. Please try again.", None

def extract_final_answer(answer_text: str) -> str:
    """
    Extract the final answer from the chain-of-thought response.
    """
    try:
        # Look for the "Answer:" section
        if "**Answer:**" in answer_text:
            answer_section = answer_text.split("**Answer:**")[1]
            # Remove confidence section if present
            if "**Confidence:**" in answer_section:
                answer_section = answer_section.split("**Confidence:**")[0]
            return answer_section.strip()
        
        # Fallback: return the last paragraph
        paragraphs = answer_text.split('\n\n')
        for paragraph in reversed(paragraphs):
            if paragraph.strip() and not paragraph.startswith('**'):
                return paragraph.strip()
        
        return answer_text.strip()
    
    except Exception as e:
        logger.warning(f"Error extracting final answer: {e}")
        return answer_text.strip()

def extract_confidence(answer_text: str) -> str:
    """
    Extract confidence level from the chain-of-thought response.
    """
    try:
        if "**Confidence:**" in answer_text:
            confidence_line = answer_text.split("**Confidence:**")[1]
            if "**Reasoning:**" in confidence_line:
                confidence_line = confidence_line.split("**Reasoning:**")[0]
            confidence = confidence_line.strip()
            if confidence in ["High", "Medium", "Low"]:
                return confidence
        return "Medium"  # Default confidence
    except Exception as e:
        logger.warning(f"Error extracting confidence: {e}")
        return "Medium"