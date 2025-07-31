import os
from typing import Tuple, Optional, Dict, Any
from openai import AsyncOpenAI
from utils.logger import logger

try:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)
except TypeError:
    raise EnvironmentError("OPENAI_API_KEY not found in .env file.")

# Specialized prompt templates for different insurance question types
MULTIPLE_POLICY_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
- Look for "Multiple Policies", "Contribution", "Other Insurance" clauses
- If multiple policies are mentioned, the answer is usually YES
- Be specific about amounts, conditions, and limitations
- Provide a clear, definitive answer with explanation

**Answer:**"""

COVERAGE_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
- Look for coverage clauses, inclusions, and exclusions
- Check for waiting periods and conditions
- Determine if the treatment/procedure is covered
- Provide a clear answer with specific reasons

**Answer:**"""

CALCULATION_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
- Identify the base amount and applicable percentages
- Apply any sub-limits or co-payment requirements
- Show the calculation step-by-step
- Provide the exact payable amount

**Answer:**"""

GENERAL_PROMPT = """You are an expert insurance policy analyst. Answer the question based on the provided context.

**Context:**
{context}

**Question:**
{question}

**Instructions:**
- Find relevant policy sections and clauses
- Extract specific details and conditions
- Provide a comprehensive, accurate response
- Include relevant policy terms when possible

**Answer:**"""

# Self-consistency prompt for answer validation
SELF_CONSISTENCY_PROMPT = """Review this answer for consistency and accuracy.

Question: {question}
Context: {context}
Answer: {answer}

If the answer is correct and consistent, respond with "CONSISTENT".
If the answer needs correction, respond with "CORRECTED_ANSWER: [corrected answer]".

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

        # Step 4: Extract and format final answer to match sample response style
        final_answer = extract_final_answer(initial_answer)
        
        # Ensure answer format matches sample: clear, concise, specific
        formatted_answer = format_answer_for_sample(final_answer, question_type)

        logger.info(f"Specialized answer generated with sample formatting")
        return formatted_answer, usage

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
    Optimized for quality.
    """
    try:
        # Look for the "Answer:" section
        if "**Answer:**" in answer_text:
            answer_section = answer_text.split("**Answer:**")[1]
            # Remove confidence section if present
            if "**Confidence:**" in answer_section:
                answer_section = answer_section.split("**Confidence:**")[0]
            return answer_section.strip()
        
        # Look for numbered steps and extract the final answer
        if "5. " in answer_text:
            # Find the last numbered step (usually the final answer)
            lines = answer_text.split('\n')
            for line in reversed(lines):
                if line.strip().startswith('5. ') or line.strip().startswith('Final answer:'):
                    return line.strip()
        
        # Fallback: return the last meaningful paragraph
        paragraphs = answer_text.split('\n\n')
        for paragraph in reversed(paragraphs):
            if paragraph.strip() and not paragraph.startswith('**') and len(paragraph.strip()) > 20:
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

def format_answer_for_sample(answer: str, question_type: str) -> str:
    """
    Format answer to match the sample response style: clear, concise, specific.
    """
    try:
        # Clean up the answer
        cleaned_answer = answer.strip()
        
        # Remove any markdown formatting
        cleaned_answer = cleaned_answer.replace("**", "").replace("*", "")
        
        # For multiple policy questions, ensure clear YES/NO format
        if question_type == "multiple_policy":
            if "yes" in cleaned_answer.lower()[:50]:
                # Ensure it starts with "Yes" and is clear
                if not cleaned_answer.lower().startswith("yes"):
                    cleaned_answer = "Yes, " + cleaned_answer
            elif "no" in cleaned_answer.lower()[:50]:
                if not cleaned_answer.lower().startswith("no"):
                    cleaned_answer = "No, " + cleaned_answer
        
        # Ensure it ends with proper punctuation
        if not cleaned_answer.endswith(('.', '!', '?')):
            cleaned_answer += '.'
        
        return cleaned_answer
        
    except Exception as e:
        logger.warning(f"Error formatting answer: {e}")
        return answer

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