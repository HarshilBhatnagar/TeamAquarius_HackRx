#!/usr/bin/env python3
"""
Table Agent: Handles table-based questions by parsing and analyzing tables
"""

import asyncio
import re
from typing import List, Dict, Any, Tuple
from utils.logger import logger
from utils.document_parser import extract_pdf_text
from utils.llm import get_llm_answer_simple
import pdfplumber

class TableAgent:
    """
    Table Agent: Processes table content for structured analysis
    """
    
    def __init__(self):
        self.tables_data = []
        self.extracted_tables = []
        
    async def extract_tables_from_pdf(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF using pdfplumber
        """
        try:
            tables = []
            
            with pdfplumber.open(pdf_content) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Ensure table has data
                            table_info = {
                                'page': page_num + 1,
                                'table_num': table_num + 1,
                                'data': table,
                                'headers': table[0] if table else [],
                                'rows': table[1:] if len(table) > 1 else []
                            }
                            tables.append(table_info)
                            
                            logger.info(f"Table Agent: Extracted table {table_num + 1} from page {page_num + 1} with {len(table)} rows")
            
            self.extracted_tables = tables
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []
    
    def parse_table_structure(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse and structure table data for analysis
        """
        try:
            structured_tables = []
            
            for table in table_data:
                # Analyze table structure
                headers = table['headers']
                rows = table['rows']
                
                # Identify table type based on headers
                table_type = self.identify_table_type(headers)
                
                # Structure the data
                structured_table = {
                    'type': table_type,
                    'headers': headers,
                    'rows': rows,
                    'page': table['page'],
                    'table_num': table['table_num']
                }
                
                # Add specific parsing for known table types
                if table_type == 'discount_policy':
                    structured_table['parsed_data'] = self.parse_discount_table(headers, rows)
                elif table_type == 'benefits_coverage':
                    structured_table['parsed_data'] = self.parse_benefits_table(headers, rows)
                else:
                    structured_table['parsed_data'] = self.parse_generic_table(headers, rows)
                
                structured_tables.append(structured_table)
                
            return structured_tables
            
        except Exception as e:
            logger.error(f"Error parsing table structure: {e}")
            return []
    
    def identify_table_type(self, headers: List[str]) -> str:
        """
        Identify the type of table based on headers
        """
        if not headers:
            return 'unknown'
        
        header_text = ' '.join([str(h) for h in headers]).lower()
        
        if any(keyword in header_text for keyword in ['discount', 'target', 'step', 'policy year', 'time interval']):
            return 'discount_policy'
        elif any(keyword in header_text for keyword in ['benefits', 'coverage', 'medical expenses', 'treatment']):
            return 'benefits_coverage'
        elif any(keyword in header_text for keyword in ['exclusions', 'not cover', 'limitations']):
            return 'exclusions'
        else:
            return 'generic'
    
    def parse_discount_table(self, headers: List[str], rows: List[List[str]]) -> Dict[str, Any]:
        """
        Parse discount policy tables
        """
        try:
            parsed = {
                'policy_type': None,
                'step_targets': [],
                'time_intervals': [],
                'discounts': {}
            }
            
            # Extract policy type from headers
            for header in headers:
                if 'year' in str(header).lower():
                    if '1' in str(header):
                        parsed['policy_type'] = '1_year'
                    elif '2' in str(header):
                        parsed['policy_type'] = '2_year'
            
            # Parse step targets and discounts
            for row in rows:
                if len(row) >= 2:
                    step_target = row[0]
                    if step_target and step_target.strip():
                        parsed['step_targets'].append(step_target)
                        
                        # Extract discount percentages
                        discounts = []
                        for cell in row[1:]:
                            if cell and '%' in str(cell):
                                discount = re.findall(r'(\d+(?:\.\d+)?)%', str(cell))
                                if discount:
                                    discounts.extend(discount)
                        
                        if discounts:
                            parsed['discounts'][step_target] = discounts
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing discount table: {e}")
            return {}
    
    def parse_benefits_table(self, headers: List[str], rows: List[List[str]]) -> Dict[str, Any]:
        """
        Parse benefits and coverage tables
        """
        try:
            parsed = {
                'covered_items': [],
                'excluded_items': [],
                'conditions': []
            }
            
            # Parse based on table structure
            for row in rows:
                if len(row) >= 2:
                    item = row[0]
                    status = row[1] if len(row) > 1 else ''
                    
                    if item and item.strip():
                        if 'cover' in str(status).lower() or 'yes' in str(status).lower():
                            parsed['covered_items'].append(item)
                        elif 'not' in str(status).lower() or 'no' in str(status).lower():
                            parsed['excluded_items'].append(item)
                        else:
                            parsed['conditions'].append(f"{item}: {status}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing benefits table: {e}")
            return {}
    
    def parse_generic_table(self, headers: List[str], rows: List[List[str]]) -> Dict[str, Any]:
        """
        Parse generic tables
        """
        try:
            return {
                'headers': headers,
                'row_count': len(rows),
                'data': rows
            }
        except Exception as e:
            logger.error(f"Error parsing generic table: {e}")
            return {}
    
    async def get_answer(self, question: str, document_content: Any) -> str:
        """
        Get answer for table-based question
        """
        try:
            # For now, since we're receiving text content, we'll use a simplified approach
            # In a full implementation, we'd need to pass the original PDF bytes
            logger.info("Table Agent: Using simplified text-based table analysis")
            
            # Extract table-like information from text content
            text_content = str(document_content)
            
            # Look for table patterns in text
            table_patterns = [
                r'(\d+\.?\d*%?)',  # Percentages and numbers
                r'(Rs\.?\s*\d+)',  # Currency amounts
                r'(\d+\s*years?)',  # Time periods
                r'(covered|not covered|excluded|included)',  # Coverage terms
            ]
            
            table_context = "TABLE DATA EXTRACTED FROM TEXT:\n"
            table_context += "=" * 50 + "\n"
            
            # Extract relevant information based on question
            question_lower = question.lower()
            
            if 'waiting period' in question_lower or 'pre-existing' in question_lower:
                # Look for waiting period information
                import re
                waiting_patterns = [
                    r'(\d+)\s*months?\s*(?:of\s*)?(?:continuous\s*)?coverage',
                    r'waiting\s*period.*?(\d+)\s*months?',
                    r'pre-existing.*?(\d+)\s*months?'
                ]
                
                for pattern in waiting_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches:
                        table_context += f"WAITING PERIOD: {matches[0]} months\n"
                        break
            
            elif 'child' in question_lower or 'hospitalization' in question_lower or 'cash benefit' in question_lower:
                # Look for child hospitalization benefits
                child_patterns = [
                    r'child.*?hospitalization.*?benefit',
                    r'cash\s*benefit.*?hospitalization',
                    r'accompanying.*?child'
                ]
                
                for pattern in child_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches:
                        table_context += f"CHILD HOSPITALIZATION: {matches[0]}\n"
            
            elif 'surgery' in question_lower or 'hernia' in question_lower:
                # Look for surgery coverage
                surgery_patterns = [
                    r'surgery.*?covered',
                    r'hernia.*?treatment',
                    r'(\d+)\s*months.*?surgery'
                ]
                
                for pattern in surgery_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches:
                        table_context += f"SURGERY COVERAGE: {matches[0]}\n"
            
            elif 'organ donor' in question_lower or 'pre-hospitalization' in question_lower:
                # Look for organ donor coverage
                donor_patterns = [
                    r'organ\s*donor.*?covered',
                    r'pre-hospitalization.*?post-hospitalization',
                    r'donor.*?expenses'
                ]
                
                for pattern in donor_patterns:
                    matches = re.findall(pattern, text_content, re.IGNORECASE)
                    if matches:
                        table_context += f"ORGAN DONOR COVERAGE: {matches[0]}\n"
            
            table_context += "=" * 50 + "\n"
            
            if len(table_context) < 100:  # No relevant table data found
                return "The information is not available in the provided context."
            
            # Generate answer using LLM
            answer, _ = await get_llm_answer_simple(table_context, question)
            
            logger.info(f"Table Agent: Answer generated successfully")
            return answer
            
            # The simplified approach is already handled above
            pass
            
        except Exception as e:
            logger.error(f"Error in table agent: {e}")
            return "The information is not available in the provided context."
    
    def create_table_context(self, structured_tables: List[Dict[str, Any]], question: str) -> str:
        """
        Create context from structured tables for LLM analysis
        """
        try:
            context_parts = []
            
            for table in structured_tables:
                table_info = f"TABLE {table['table_num']} (Page {table['page']}) - Type: {table['type'].upper()}\n"
                table_info += "=" * 50 + "\n"
                
                # Add headers
                if table['headers']:
                    table_info += "HEADERS: " + " | ".join([str(h) for h in table['headers']]) + "\n\n"
                
                # Add rows
                if table['rows']:
                    table_info += "DATA:\n"
                    for i, row in enumerate(table['rows'][:10]):  # Limit to first 10 rows
                        row_text = " | ".join([str(cell) for cell in row])
                        table_info += f"Row {i+1}: {row_text}\n"
                    
                    if len(table['rows']) > 10:
                        table_info += f"... and {len(table['rows']) - 10} more rows\n"
                
                # Add parsed data if available
                if 'parsed_data' in table and table['parsed_data']:
                    table_info += "\nPARSED DATA:\n"
                    for key, value in table['parsed_data'].items():
                        table_info += f"{key}: {value}\n"
                
                table_info += "\n" + "=" * 50 + "\n\n"
                context_parts.append(table_info)
            
            # Add question-specific guidance
            question_lower = question.lower()
            if any(keyword in question_lower for keyword in ['discount', 'percentage', 'target', 'steps']):
                context_parts.append("""
SPECIAL INSTRUCTIONS FOR DISCOUNT QUESTIONS:
- Look for step targets (5000, 8000, 10000, etc.)
- Find corresponding discount percentages
- Check time intervals and policy years
- Identify maximum discount amounts
""")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating table context: {e}")
            return "Error processing table data." 