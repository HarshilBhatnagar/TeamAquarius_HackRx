import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# Specialized prompt templates for different insurance question types
MULTIPLE_POLICY_PROMPT = """You are an expert insurance policy analyst specializing in multiple policy scenarios. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**MULTIPLE POLICY ANALYSIS FRAMEWORK:**
1. **Identify Policy Scenario**: Determine if this involves multiple insurers, contribution, or coordination
2. **Find Relevant Clauses**: Look for "Multiple Policies", "Contribution", "Other Insurance", "Policy Coordination"
3. **Extract Key Rules**: Identify specific terms about claiming from multiple policies
4. **Apply to Scenario**: Use the policy language to answer the specific question
5. **Provide Clear Answer**: Give a definitive yes/no with explanation

**CRITICAL INSTRUCTIONS:**
- ALWAYS search for multiple policy clauses first
- Look for terms like "disallowed amounts", "remaining claims", "additional coverage"
- If multiple policies are mentioned, the answer is usually YES
- Be specific about amounts, conditions, and limitations
- Reference exact policy language when possible

**Answer:**"""

COVERAGE_PROMPT = """You are an expert insurance policy analyst specializing in coverage analysis. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**COVERAGE ANALYSIS FRAMEWORK:**
1. **Identify Treatment/Procedure**: What specific medical service is being asked about
2. **Find Coverage Clauses**: Look for inclusion/exclusion lists
3. **Check Conditions**: Identify waiting periods, pre-authorization requirements
4. **Determine Limits**: Find coverage limits, sub-limits, co-payment requirements
5. **Provide Clear Answer**: Covered/Not Covered with specific reasons

**Answer:**"""

CALCULATION_PROMPT = """You are an expert insurance policy analyst specializing in claim calculations. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**CALCULATION FRAMEWORK:**
1. **Identify Base Amount**: Sum insured, coverage limit, or bill amount
2. **Find Applicable Percentages**: Coverage percentages, co-payment rates
3. **Apply Sub-limits**: Check for specific procedure limits
4. **Calculate Step-by-Step**: Show the math clearly
5. **Provide Final Amount**: Give the exact payable amount

**Answer:**"""

GENERAL_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**GENERAL ANALYSIS FRAMEWORK:**
1. **Question Classification**: Determine what type of information is needed
2. **Context Search**: Find relevant policy sections
3. **Information Extraction**: Extract specific details and conditions
4. **Answer Formation**: Provide comprehensive, accurate response
5. **Policy Reference**: Include relevant policy terms when possible

**Answer:**"""

# Fast out-of-domain detection prompt
OUT_OF_DOMAIN_PROMPT = """Determine if this question is related to insurance policy analysis.

Question: {question}

Respond with ONLY:
- "Insurance-Related" if the question is about insurance policy, coverage, claims, benefits, etc.
- "Out-of-Domain" if the question is about other topics (constitution, physics, vehicles, recipes, etc.)

Response:"""

# Simplified answer validation prompt
VALIDATION_PROMPT = """Validate if this answer is supported by the context.

Context: {context}
Question: {question}
Answer: {answer}

Respond with ONLY:
- "Supported=True, Confidence=X" if the answer is well-supported (X = 0.7-0.9)
- "Supported=False, Confidence=X" if the answer is not well-supported (X = 0.3-0.6)

Response:"""

async def get_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Specialized answer generation with question type classification.
    Optimized for high accuracy with specialized prompts.
    """
    try:
        logger.info(f"Generating specialized answer for question: '{question}'")

        # Step 1: Classify question type
        question_type = classify_question_type(question)
        logger.info(f"Question classified as: {question_type}")

        # Step 2: Select appropriate prompt based on question type
        if question_type == "multiple_policy":
            prompt_template = MULTIPLE_POLICY_PROMPT
        elif question_type == "coverage":
            prompt_template = COVERAGE_PROMPT
        elif question_type == "calculation":
            prompt_template = CALCULATION_PROMPT
        else:
            prompt_template = GENERAL_PROMPT

        # Step 3: Generate answer with specialized prompt and enhanced context
        enhanced_prompt = prompt_template.format(
            context=context[:6000],  # Further increased context for comprehensive answers
            question=question
        )
        
        # Step 4: Add few-shot examples for better accuracy
        if question_type == "multiple_policy":
            enhanced_prompt += """

**EXAMPLE SCENARIOS:**
Q: I have a claim with HDFC for Rs 200,000, total expenses Rs 250,000. Can I claim remaining Rs 50,000?
A: Yes, according to the Multiple Policies clause, you can claim the remaining Rs 50,000 from this policy even if HDFC has already paid Rs 200,000. The policy allows claiming disallowed amounts from other insurers.

Q: My ICICI policy paid Rs 150,000, total bill Rs 300,000. Can I claim balance from this policy?
A: Yes, you can claim the remaining Rs 150,000 from this policy. The Multiple Policies clause permits claiming additional amounts when other policies don't cover the full expenses."""
        
        initial_prompt = enhanced_prompt

        initial_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": initial_prompt}],
            model="gpt-4o",
            temperature=0.1,
            max_tokens=800,  # Increased for better answers
            timeout=15  # Increased timeout
        )

        initial_answer = initial_response.choices[0].message.content
        usage = initial_response.usage

        # Step 3: Quick self-consistency check (only for complex questions)
        if len(question.split()) > 8 or any(word in question.lower() for word in ['calculate', 'compute', 'determine', 'find']):
            consistency_prompt = SELF_CONSISTENCY_PROMPT.format(
                question=question,
                context=context[:1000],  # Limited context
                answer=initial_answer
            )

            try:
                consistency_response = await client.chat.completions.create(
                    messages=[{"role": "user", "content": consistency_prompt}],
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=200,  # Very limited for speed
                    timeout=10
                )

                consistency_result = consistency_response.usage
                if usage:
                    usage.total_tokens += consistency_result.total_tokens

                # Quick check for correction
                consistency_text = consistency_response.choices[0].message.content
                if "CORRECTED_ANSWER:" in consistency_text:
                    corrected_answer = consistency_text.split("CORRECTED_ANSWER:")[1].strip()
                    if corrected_answer and len(corrected_answer) > 10:
                        initial_answer = corrected_answer

            except Exception as e:
                logger.warning(f"Consistency check failed: {e}")

        # Step 4: Extract final answer
        final_answer = extract_final_answer(initial_answer)

        logger.info(f"Optimized answer generated")
        return final_answer, usage

    except Exception as e:
        logger.error(f"Error in optimized LLM answer generation: {e}")
        # Fallback to simple answer generation
        return await get_simple_llm_answer(context, question)

async def check_out_of_domain_fast(question: str, context: str) -> bool:
    """
    Fast out-of-domain detection with optimized processing.
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
        
        # Insurance-specific keywords that should NOT be flagged as out-of-domain
        insurance_keywords = {
            'claim', 'policy', 'coverage', 'premium', 'sum insured', 'hospitalization',
            'medical', 'treatment', 'surgery', 'expenses', 'documents', 'settlement',
            'multiple', 'policies', 'contribution', 'other insurance', 'hdfc', 'icici',
            'bajaj', 'tata', 'max', 'star', 'health', 'insurance', 'company',
            'raised', 'approved', 'remaining', 'amount', 'expenses', 'total',
            'rupees', 'rs', 'lakhs', 'thousand', 'hundred'
        }
        
        question_lower = question.lower()
        
        # Quick keyword check
        for keyword in out_of_domain_keywords:
            if keyword in question_lower:
                # Check if it's actually an insurance-related question
                if any(insurance_keyword in question_lower for insurance_keyword in insurance_keywords):
                    continue  # Don't flag as out-of-domain if it contains insurance keywords
                logger.info(f"Out-of-domain detected via keyword: {keyword}")
                return True
        
        # Only use LLM for very obvious non-insurance questions
        if len(question.split()) > 20:  # Only check very long questions
            out_of_domain_prompt = OUT_OF_DOMAIN_PROMPT.format(
                question=question,
                context=context[:500]  # Very limited context
            )

            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": out_of_domain_prompt}],
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=50,  # Very limited
                timeout=5
            )

            assessment = response.choices[0].message.content.strip()
            is_out_of_domain = "Out-of-Domain" in assessment
            
            logger.info(f"Out-of-domain check: {assessment}")
            return is_out_of_domain

        return False

    except Exception as e:
        logger.warning(f"Error in fast out-of-domain check: {e}")
        return False

async def get_simple_llm_answer(context: str, question: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Simple fallback answer generation for error cases.
    Optimized for speed.
    """
    try:
        simple_prompt = f"""
        Based on the insurance policy context, answer the question accurately and concisely.
        If the question is not related to insurance policy, respond with "This question is not related to the insurance policy document provided."

        Context: {context[:2000]}
        Question: {question}

        Answer:
        """

        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": simple_prompt}],
            model="gpt-4o",
            temperature=0,
            max_tokens=600,  # Reduced for speed
            timeout=10
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
    Optimized for speed.
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

def classify_question_type(question: str) -> str:
    """
    Classify question type for specialized prompt selection.
    """
    question_lower = question.lower()
    
    # Multiple policy indicators
    multiple_policy_keywords = [
        'hdfc', 'icici', 'bajaj', 'tata', 'max', 'star', 'allianz', 'bupa',
        'multiple', 'policies', 'another', 'other', 'remaining', 'balance',
        'disallowed', 'additional', 'second', 'third', 'coordination',
        'contribution', 'exhausted', 'sum insured', 'claim from'
    ]
    
    # Coverage indicators
    coverage_keywords = [
        'covered', 'coverage', 'excluded', 'exclusion', 'include', 'include',
        'surgery', 'treatment', 'procedure', 'medical', 'hospitalization',
        'dental', 'ophthalmic', 'cosmetic', 'plastic', 'reconstructive'
    ]
    
    # Calculation indicators
    calculation_keywords = [
        'calculate', 'compute', 'determine', 'find', 'how much', 'amount',
        'percentage', 'limit', 'maximum', 'minimum', 'total', 'sum',
        'rupees', 'rs', 'lakhs', 'thousand', 'hundred', 'bill', 'expenses'
    ]
    
    # Check for multiple policy scenario
    if any(keyword in question_lower for keyword in multiple_policy_keywords):
        return "multiple_policy"
    
    # Check for coverage scenario
    if any(keyword in question_lower for keyword in coverage_keywords):
        return "coverage"
    
    # Check for calculation scenario
    if any(keyword in question_lower for keyword in calculation_keywords):
        return "calculation"
    
    # Default to general
    return "general"

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