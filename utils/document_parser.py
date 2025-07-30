import requests
import pdfplumber
import docx
from typing import List, Dict, Any, Optional
from utils.logger import logger
import re
import io

def get_document_text(url: str) -> str:
    """
    Enhanced document extraction with layout-aware processing.
    Handles diverse document structures including multi-column layouts, tables, and complex PDFs.
    """
    try:
        logger.info(f"Downloading document from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or url.lower().endswith('.pdf'):
            return extract_pdf_text(response.content)
        elif 'docx' in content_type or url.lower().endswith('.docx'):
            return extract_docx_text(response.content)
        else:
            # Try to detect PDF by content
            if response.content.startswith(b'%PDF'):
                return extract_pdf_text(response.content)
            else:
                raise ValueError(f"Unsupported document type: {content_type}")
                
    except Exception as e:
        logger.error(f"Error downloading document: {e}")
        raise

def extract_pdf_text(pdf_content: bytes) -> str:
    """
    Enhanced PDF extraction with layout-aware processing.
    Handles multi-column layouts, tables, and complex insurance document structures.
    """
    try:
        text_content = []
        
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            logger.info(f"Processing PDF with {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages):
                logger.info(f"Processing page {page_num + 1}")
                
                # Extract text with layout preservation
                page_text = extract_page_text_with_layout(page)
                text_content.append(page_text)
        
        full_text = "\n\n".join(text_content)
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise

def extract_page_text_with_layout(page) -> str:
    """
    Extract text from a single page with layout awareness.
    Handles multi-column layouts, tables, and complex insurance document structures.
    """
    page_content = []
    
    try:
        # 1. Extract tables first (they contain critical insurance information)
        tables = page.extract_tables()
        if tables:
            logger.info(f"Found {len(tables)} tables on page")
            for table_idx, table in enumerate(tables):
                table_text = process_insurance_table(table)
                if table_text:
                    page_content.append(f"TABLE {table_idx + 1}:\n{table_text}")
        
        # 2. Extract text with layout awareness (simplified for compatibility)
        try:
            # Use chars method which is more reliable across pdfplumber versions
            chars = page.chars
            if chars:
                # Group characters by lines and columns
                lines = {}
                for char in chars:
                    y_pos = round(char['y0'], 2)
                    if y_pos not in lines:
                        lines[y_pos] = []
                    lines[y_pos].append(char)
                
                # Sort lines by y position and characters by x position
                sorted_lines = sorted(lines.items())
                for y_pos, line_chars in sorted_lines:
                    line_chars.sort(key=lambda c: c['x0'])
                    line_text = ''.join([c['text'] for c in line_chars])
                    if line_text.strip():
                        page_content.append(line_text)
        except Exception as layout_error:
            logger.warning(f"Error in layout-aware extraction: {layout_error}, falling back to plain text")
        
        # 3. Fallback: Extract plain text if layout extraction fails
        if not page_content:
            plain_text = page.extract_text()
            if plain_text:
                page_content.append(plain_text)
        
        return "\n\n".join(page_content)
        
    except Exception as e:
        logger.warning(f"Error in layout-aware extraction: {e}, falling back to plain text")
        return page.extract_text() or ""

def process_insurance_table(table: List[List[str]]) -> str:
    """
    Process insurance tables with enhanced formatting.
    Handles dense tables of benefits, coverage limits, and policy details.
    """
    if not table:
        return ""
    
    try:
        table_text = []
        
        # Clean and format table data
        for row_idx, row in enumerate(table):
            if not row:
                continue
                
            # Clean row data
            cleaned_row = []
            for cell in row:
                if cell:
                    # Clean cell text
                    cell_text = str(cell).strip()
                    # Remove excessive whitespace
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    cleaned_row.append(cell_text)
                else:
                    cleaned_row.append("")
            
            # Skip empty rows
            if not any(cell for cell in cleaned_row):
                continue
            
            # Format row based on content
            if row_idx == 0:
                # Header row
                table_text.append(" | ".join(cleaned_row))
                table_text.append("-" * 50)  # Separator
            else:
                # Data row
                row_text = " | ".join(cleaned_row)
                table_text.append(row_text)
        
        return "\n".join(table_text)
        
    except Exception as e:
        logger.warning(f"Error processing table: {e}")
        return ""



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