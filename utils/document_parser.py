import requests
import pdfplumber
import docx
from typing import List, Dict, Any, Optional
from utils.logger import logger
import re
import io

import asyncio
import aiohttp

async def get_document_text(url: str) -> str:
    """
    Simple document extraction - get everything and let LLM handle it
    """
    try:
        logger.info(f"Downloading document from: {url}")
        
        # Use aiohttp for async download
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                content = await response.read()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            text = extract_pdf_text(content)
            return text
        elif 'docx' in content_type or url.lower().endswith('.docx'):
            text = extract_docx_text(content)
            return text
        else:
            # Try to detect PDF by content
            if content.startswith(b'%PDF'):
                text = extract_pdf_text(content)
                return text
            else:
                raise ValueError(f"Unsupported document type: {content_type}")
                
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise

def extract_pdf_text(pdf_content: bytes) -> str:
    """
    Simple PDF extraction - get all text without aggressive filtering
    """
    try:
        text_content = []
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            # Process all pages (no limit)
            for page_num in range(total_pages):
                page = pdf.pages[page_num]
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                # Extract text simply
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                
                # Extract tables as text
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        if table:
                            table_text = format_table_simple(table)
                            text_content.append(f"\nTABLE {table_idx + 1}:\n{table_text}\n")
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        
        # Minimal cleaning only
        full_text = clean_document_content_minimal(full_text)
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise

def format_table_simple(table_data):
    """
    Simple table formatting - just join with pipes
    """
    if not table_data:
        return ""
    
    try:
        table_lines = []
        for row in table_data:
            if row:
                # Clean and join row
                cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                table_lines.append(" | ".join(cleaned_row))
        
        return "\n".join(table_lines)
        
    except Exception as e:
        logger.warning(f"Error formatting table: {e}")
        return str(table_data)

def clean_document_content_minimal(text: str) -> str:
    """
    Minimal cleaning - only remove obvious headers/footers
    """
    try:
        lines = text.split('\n')
        cleaned_lines = []
        
        # Only skip obvious page numbers and headers
        skip_patterns = [
            r'^\s*\d+\s*\|\s*Page',  # Page numbers
            r'^\s*Page\s+\d+',  # Page numbers
            r'^\s*©\s*\d+',  # Copyright
            r'^\s*www\.',  # Website URLs
            r'^\s*https?://',  # URLs
        ]
        
        import re
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip only obvious headers/footers
            skip_line = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip_line = True
                    break
            
            if not skip_line:
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
        
        logger.info(f"Minimal cleaning: {len(text)} -> {len(cleaned_text)} characters")
        return cleaned_text
        
    except Exception as e:
        logger.warning(f"Error in minimal cleaning: {e}")
        return text

def extract_docx_text(docx_content: bytes) -> str:
    """
    Extract text from DOCX documents.
    """
    try:
        doc = docx.Document(io.BytesIO(docx_content))
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text)
        
        # Extract tables from DOCX
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    table_text.append(" | ".join(row_text))
            
            if table_text:
                text_content.append("TABLE:\n" + "\n".join(table_text))
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Extracted {len(full_text)} characters from DOCX")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise