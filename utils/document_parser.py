import io
import re
import docx
import requests
import pdfplumber
from urllib.parse import urlparse
from .logger import logger

def _format_table(table_data):
    """Converts a list of lists from pdfplumber into a clean Markdown table string."""
    if not table_data:
        return ""

    # Clean up each cell by replacing None with "" and stripping whitespace/newlines
    cleaned_table = [[(str(cell) or "").replace("\n", " ").strip() for cell in row] for row in table_data]

    # Create the header row
    header = " | ".join(cleaned_table[0])
    markdown_table = f"| {header} |\n"
    
    # Create the separator row
    separator = " | ".join(["---"] * len(cleaned_table[0]))
    markdown_table += f"| {separator} |\n"
    
    # Create the data rows
    for row in cleaned_table[1:]:
        # Ensure row has the same number of columns as the header
        while len(row) < len(cleaned_table[0]):
            row.append("")
        data_row = " | ".join(row)
        markdown_table += f"| {data_row} |\n"
        
    return markdown_table

def get_document_text(url: str) -> str:
    """
    Downloads a document and extracts clean text content.
    Uses an advanced method for PDFs to identify and format tables as Markdown.
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
        try:
            with pdfplumber.open(file_content) as pdf:
                for page in pdf.pages:
                    # Extract plain text from the page
                    page_text = page.extract_text()
                    if page_text:
                        full_text_parts.append(page_text)
                    
                    # Extract tables and format them as Markdown
                    tables = page.extract_tables()
                    for table in tables:
                        if table: # Ensure the table is not empty
                            markdown_table = _format_table(table)
                            full_text_parts.append(f"\n\n--- TABLE START ---\n{markdown_table}\n--- TABLE END ---\n\n")
        except Exception as e:
            logger.error(f"Failed to parse PDF with pdfplumber: {e}")
            raise RuntimeError(f"Could not parse PDF: {e}")
        
        text = "\n".join(full_text_parts)

    elif file_path.lower().endswith('.docx'):
        logger.info("Parsing DOCX document.")
        doc = docx.Document(file_content)
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported document type.")

    logger.info(f"Successfully extracted {len(text)} characters. Starting text cleaning.")
    text = re.sub(r'\s+', ' ', text).strip()
    logger.info(f"Text cleaning complete. Final length: {len(text)} characters.")
    
    return text