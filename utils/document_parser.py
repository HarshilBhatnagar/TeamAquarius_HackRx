import io
import re
import docx
import requests
import pdfplumber
from urllib.parse import urlparse
from .logger import logger

def _format_table(table_data):
    """Converts a list of lists into a Markdown table string."""
    if not table_data:
        return ""

    markdown_table = ""
    # Create header row
    header = " | ".join(str(cell or "").replace("\n", " ") for cell in table_data[0])
    markdown_table += f"| {header} |\n"
    
    # Create separator row
    separator = " | ".join(["---"] * len(table_data[0]))
    markdown_table += f"| {separator} |\n"
    
    # Create data rows
    for row in table_data[1:]:
        # Ensure row has the same number of columns as the header
        while len(row) < len(table_data[0]):
            row.append(None)
        data_row = " | ".join(str(cell or "").replace("\n", " ") for cell in row)
        markdown_table += f"| {data_row} |\n"
        
    return markdown_table

def get_document_text(url: str) -> str:
    """
    Downloads a document, extracts text and tables, and formats tables as Markdown.
    """
    url_str = str(url)
    parsed_url = urlparse(url_str)
    file_path = parsed_url.path

    logger.info(f"Downloading document from {url_str}")
    try:
        response = requests.get(url_str, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download document: {e}")
        raise

    file_content = io.BytesIO(response.content)
    text = ""

    if file_path.lower().endswith('.pdf'):
        logger.info("Parsing PDF with table-aware logic using pdfplumber.")
        full_text_parts = []
        with pdfplumber.open(file_content) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                tables = page.extract_tables()
                
                if page_text:
                    full_text_parts.append(page_text)
                
                for table in tables:
                    if table:
                        markdown_table = _format_table(table)
                        full_text_parts.append(f"\n\n{markdown_table}\n\n")
        text = "\n".join(full_text_parts)

    elif file_path.lower().endswith('.docx'):
        logger.info("Parsing DOCX document.")
        doc = docx.Document(file_content)
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported document type. Only PDF and DOCX are supported.")

    logger.info(f"Successfully extracted {len(text)} characters. Starting text cleaning.")
    text = re.sub(r'\s+', ' ', text).strip()
    logger.info(f"Text cleaning complete. Final length: {len(text)} characters.")
    
    return text