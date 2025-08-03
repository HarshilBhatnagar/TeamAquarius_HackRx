#!/usr/bin/env python3
"""
Master Agent: Orchestrates between Text and Table agents
Uses Chain-of-Thought reasoning to synthesize final responses
"""

import asyncio
from typing import List, Dict, Any, Tuple
from utils.logger import logger
from services.text_agent import TextAgent
from services.table_agent import TableAgent
from utils.llm import get_llm_answer_simple

class MasterAgent:
    """
    Master Agent: Orchestrates between specialized agents
    """
    
    def __init__(self):
        self.text_agent = TextAgent()
        self.table_agent = TableAgent()
        
    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze question to determine which agents to call
        """
        question_lower = question.lower()
        
        # Keywords that suggest table-based information
        table_keywords = [
            'discount', 'percentage', 'target', 'steps', 'policy year', 
            'time interval', 'maximum discount', 'average step',
            '5000', '8000', '10000', '90 days', '180 days', '270 days',
            '300 days', '360 days', '450 days', '540 days', '630 days', '660 days'
        ]
        
        # Keywords that suggest text-based information
        text_keywords = [
            'waiting period', 'pre-existing', 'coverage', 'benefits',
            'exclusions', 'hospitalization', 'child', 'hernia', 'surgery',
            'organ donor', 'medical expenses', 'treatment', 'policy terms'
        ]
        
        # Count matches
        table_score = sum(1 for keyword in table_keywords if keyword in question_lower)
        text_score = sum(1 for keyword in text_keywords if keyword in question_lower)
        
        # Decision logic
        if table_score > text_score:
            return {
                'primary_agent': 'table',
                'secondary_agent': 'text',
                'reasoning': f'Question contains {table_score} table-related keywords vs {text_score} text keywords'
            }
        elif text_score > table_score:
            return {
                'primary_agent': 'text', 
                'secondary_agent': 'table',
                'reasoning': f'Question contains {text_score} text-related keywords vs {table_score} table keywords'
            }
        else:
            return {
                'primary_agent': 'both',
                'secondary_agent': None,
                'reasoning': f'Equal scores: {text_score} text keywords, {table_score} table keywords'
            }
    
    async def get_agent_response(self, agent_type: str, question: str, document_content: Any) -> str:
        """
        Get response from specific agent
        """
        try:
            if agent_type == 'text':
                return await self.text_agent.get_answer(question, document_content)
            elif agent_type == 'table':
                return await self.table_agent.get_answer(question, document_content)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            logger.error(f"Error getting response from {agent_type} agent: {e}")
            return f"Error in {agent_type} agent processing"
    
    async def synthesize_response(self, question: str, responses: Dict[str, str]) -> str:
        """
        Use Chain-of-Thought to synthesize final response from multiple agents
        """
        try:
            # Create synthesis prompt
            synthesis_prompt = f"""
You are a master insurance policy analyst. Synthesize the following responses from specialized agents into a single, comprehensive answer.

**QUESTION:** {question}

**AGENT RESPONSES:**
"""
            
            for agent, response in responses.items():
                synthesis_prompt += f"\n**{agent.upper()} AGENT:** {response}\n"
            
            synthesis_prompt += """
**TASK:**
1. Analyze all agent responses
2. Identify the most relevant and accurate information
3. Combine information if multiple agents have relevant data
4. Provide a single, clear, comprehensive answer
5. If one agent has the complete answer, use that
6. If multiple agents have partial information, synthesize them
7. If no agent has relevant information, state "The information is not available in the provided context."

**FINAL ANSWER:**"""
            
            # Get synthesized response
            response, _ = await get_llm_answer_simple(synthesis_prompt, question)
            return response
            
        except Exception as e:
            logger.error(f"Error in response synthesis: {e}")
            # Fallback: return the best available response
            for agent, response in responses.items():
                if "not available" not in response.lower():
                    return response
            return "The information is not available in the provided context."
    
    async def process_question(self, question: str, document_content: Any) -> str:
        """
        Main processing method: analyze question, call appropriate agents, synthesize response
        """
        try:
            logger.info(f"Master Agent: Processing question: '{question}'")
            
            # Step 1: Analyze question
            analysis = await self.analyze_question(question)
            logger.info(f"Question analysis: {analysis}")
            
            # Step 2: Call appropriate agents
            responses = {}
            
            if analysis['primary_agent'] == 'both':
                # Call both agents in parallel
                text_task = self.get_agent_response('text', question, document_content)
                table_task = self.get_agent_response('table', question, document_content)
                
                text_response, table_response = await asyncio.gather(text_task, table_task)
                responses['text'] = text_response
                responses['table'] = table_response
                
            else:
                # Call primary agent first
                primary_response = await self.get_agent_response(analysis['primary_agent'], question, document_content)
                responses[analysis['primary_agent']] = primary_response
                
                # If primary agent doesn't have answer, try secondary
                if "not available" in primary_response.lower() and analysis['secondary_agent']:
                    secondary_response = await self.get_agent_response(analysis['secondary_agent'], question, document_content)
                    responses[analysis['secondary_agent']] = secondary_response
            
            # Step 3: Synthesize final response
            final_response = await self.synthesize_response(question, responses)
            
            logger.info(f"Master Agent: Final response generated")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in master agent processing: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try again." 