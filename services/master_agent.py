#!/usr/bin/env python3
"""
Simple Master Agent: Just use text agent for all questions
"""

import asyncio
from typing import List, Dict, Any, Tuple
from utils.logger import logger
from services.text_agent import TextAgent

class MasterAgent:
    def __init__(self):
        self.text_agent = TextAgent()

    async def process_question(self, question: str, document_content: Any) -> str:
        """
        Simple processing: just use text agent for everything
        """
        try:
            logger.info(f"Master Agent: Processing question: '{question}'")
            
            # Just use text agent for all questions
            answer = await self.text_agent.get_answer(question, document_content)
            
            logger.info(f"Master Agent: Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error in master agent: {e}")
            return "The information is not available in the provided context." 