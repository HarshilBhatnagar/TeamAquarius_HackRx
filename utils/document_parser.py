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
        
        # 2. Extract text with layout awareness
        text_objects = page.extract_text_objects()
        if text_objects:
            # Sort text objects by position (top to bottom, left to right)
            sorted_objects = sort_text_objects_by_layout(text_objects)
            
            # Group objects by columns
            columns = group_objects_by_columns(sorted_objects, page.width)
            
            # Extract text from each column
            for col_idx, column in enumerate(columns):
                column_text = extract_column_text(column)
                if column_text.strip():
                    page_content.append(f"COLUMN {col_idx + 1}:\n{column_text}")
        
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

def sort_text_objects_by_layout(text_objects: List[Dict]) -> List[Dict]:
    """
    Sort text objects by their position in the document layout.
    """
    try:
        # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
        return sorted(text_objects, key=lambda obj: (obj['top'], obj['x0']))
    except Exception as e:
        logger.warning(f"Error sorting text objects: {e}")
        return text_objects

def group_objects_by_columns(text_objects: List[Dict], page_width: float) -> List[List[Dict]]:
    """
    Group text objects into columns based on their x-coordinates.
    Handles multi-column layouts common in insurance documents.
    """
    try:
        if not text_objects:
            return []
        
        # Calculate column boundaries
        x_coordinates = [obj['x0'] for obj in text_objects]
        x_coordinates.sort()
        
        # Detect column boundaries (gaps in x-coordinates)
        column_boundaries = detect_column_boundaries(x_coordinates, page_width)
        
        # Group objects by columns
        columns = [[] for _ in range(len(column_boundaries) + 1)]
        
        for obj in text_objects:
            column_idx = find_column_for_object(obj, column_boundaries)
            if column_idx < len(columns):
                columns[column_idx].append(obj)
        
        return [col for col in columns if col]
        
    except Exception as e:
        logger.warning(f"Error grouping objects by columns: {e}")
        return [text_objects]

def detect_column_boundaries(x_coordinates: List[float], page_width: float) -> List[float]:
    """
    Detect column boundaries based on gaps in x-coordinates.
    """
    try:
        if len(x_coordinates) < 2:
            return []
        
        boundaries = []
        gap_threshold = page_width * 0.1  # 10% of page width
        
        for i in range(len(x_coordinates) - 1):
            gap = x_coordinates[i + 1] - x_coordinates[i]
            if gap > gap_threshold:
                boundaries.append(x_coordinates[i] + gap / 2)
        
        return boundaries
        
    except Exception as e:
        logger.warning(f"Error detecting column boundaries: {e}")
        return []

def find_column_for_object(obj: Dict, boundaries: List[float]) -> int:
    """
    Find which column an object belongs to based on its x-coordinate.
    """
    try:
        x_center = obj['x0']
        
        for i, boundary in enumerate(boundaries):
            if x_center < boundary:
                return i
        
        return len(boundaries)
        
    except Exception as e:
        logger.warning(f"Error finding column for object: {e}")
        return 0

def extract_column_text(column_objects: List[Dict]) -> str:
    """
    Extract text from a column of text objects.
    """
    try:
        # Sort objects by y-coordinate within the column
        sorted_objects = sorted(column_objects, key=lambda obj: obj['top'])
        
        # Extract text from each object
        texts = []
        for obj in sorted_objects:
            if 'text' in obj and obj['text'].strip():
                texts.append(obj['text'].strip())
        
        return "\n".join(texts)
        
    except Exception as e:
        logger.warning(f"Error extracting column text: {e}")
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