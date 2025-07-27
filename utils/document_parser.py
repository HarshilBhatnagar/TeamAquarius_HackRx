import io
import re
import fitz
import docx
import requests
from urllib.parse import urlparse
from .logger import logger

def get_document_text(url: str) -> str:
    """
    Downloads a document from a URL, extracts, and cleans its text content.
    Supports PDF and DOCX formats.
    """
    url_str = str(url)
    
    # Parse the URL to get the path and ignore query parameters
    parsed_url = urlparse(url_str)
    file_path = parsed_url.path

    logger.info(f"Downloading document from {url_str}")
    try:
        response = requests.get(url_str, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download document: {e}")
        raise

    file_content = io.BytesIO(response.content)
    text = ""

    # Check the file extension on the path, not the full URL
    if file_path.lower().endswith('.pdf'):
        logger.info("Parsing PDF document.")
        with fitz.open(stream=file_content, filetype="pdf") as pdf_doc:
            text = "".join(page.get_text() for page in pdf_doc)
    elif file_path.lower().endswith('.docx'):
        logger.info("Parsing DOCX document.")
        doc = docx.Document(file_content)
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported document type. Only PDF and DOCX are supported.")

    logger.info(f"Successfully extracted {len(text)} characters. Starting text cleaning.")

    # --- New Cleaning & Pre-processing Logic ---
    # Normalize all whitespace (newlines, tabs, multiple spaces) into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common repeating headers/footers to create cleaner chunks
    text = re.sub(r'UIN: \w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'CIN: \w+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'IRDAI Regn\. No\.\s*–?\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # --- End of New Logic ---

    logger.info(f"Text cleaning complete. Final length: {len(text)} characters.")
    return text