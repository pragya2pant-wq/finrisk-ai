"""
Document Intelligence Service.

Handles multi-format PDF parsing, raw text extraction, and tabular financial data parsing.
Author: Pragya Pant
Institute: iPEC Solutions
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pdfplumber
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import logger
from app.schemas.document import ExtractedTable, DocumentExtractionResponse


class PDFProcessor:
    """
    Service for parsing financial PDF documents, extracting plain text for RAG,
    and capturing structured tables for ratio extraction.
    """

    @staticmethod
    def extract_text_and_tables(file_path: str) -> DocumentExtractionResponse:
        """
        Extracts both structured tables and raw text from a PDF file.

        Args:
            file_path (str): Local path to the uploaded PDF file.

        Returns:
            DocumentExtractionResponse: Processed document details and tables.
        """
        logger.info(f"Beginning PDF extraction for: {file_path}")
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Target document not found at: {file_path}")

        extracted_text_chunks: List[str] = []
        extracted_tables: List[ExtractedTable] = []

        # Use PyPDF for fast metadata and total page count
        pypdf_reader = PdfReader(str(path))
        total_pages = len(pypdf_reader.pages)

        # Use pdfplumber for high-fidelity table extraction
        with pdfplumber.open(str(path)) as pdf:
            for idx, page in enumerate(pdf.pages):
                page_num = idx + 1
                
                # Extract text
                page_text = page.extract_text() or ""
                cleaned_page_text = PDFProcessor._clean_text(page_text)
                if cleaned_page_text:
                    extracted_text_chunks.append(f"--- Page {page_num} ---\n{cleaned_page_text}")

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        # First row treated as header, subsequent rows as data
                        headers = [str(cell).strip() if cell else "" for cell in table[0]]
                        rows = [
                            [str(cell).strip() if cell else "" for cell in row]
                            for row in table[1:]
                        ]
                        extracted_tables.append(
                            ExtractedTable(
                                page_number=page_num,
                                headers=headers,
                                rows=rows
                            )
                        )

        full_text = "\n\n".join(extracted_text_chunks)
        logger.info(f"Successfully processed {total_pages} pages, extracted {len(extracted_tables)} tables.")

        return DocumentExtractionResponse(
            filename=path.name,
            total_pages=total_pages,
            raw_text=full_text,
            extracted_tables=extracted_tables,
            metadata={"file_size_bytes": path.stat().st_size}
        )

    @staticmethod
    def create_text_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        Splits extracted text into recursive character chunks with overlap for RAG retrieval.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        return splitter.split_text(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Removes excess whitespace, non-printable characters, and normalizes line breaks.
        """
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()