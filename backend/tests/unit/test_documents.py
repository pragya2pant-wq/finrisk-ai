"""
Unit and Integration Tests for PDF Processing and Document Ingestion API.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import io
import pytest
from fastapi.testclient import TestClient
from app.services.pdf_processor import PDFProcessor


def test_pdf_processor_clean_text() -> None:
    """Verify text normalization removes excess whitespace and standardizes line breaks."""
    raw_input = "Header   Text\r\n\r\nBody   content line."
    cleaned = PDFProcessor._clean_text(raw_input)
    assert cleaned == "Header Text\n\nBody content line."


def test_pdf_processor_create_text_chunks() -> None:
    """Verify recursive character text chunking splits long strings properly."""
    long_text = "Financial sentence one. " * 50
    chunks = PDFProcessor.create_text_chunks(long_text, chunk_size=200, chunk_overlap=20)
    
    assert len(chunks) > 1
    assert isinstance(chunks[0], str)


def test_pdf_processor_file_not_found() -> None:
    """Verify FileNotFoundError is raised when attempting to process a non-existent PDF."""
    with pytest.raises(FileNotFoundError):
        PDFProcessor.extract_text_and_tables("non_existent_file.pdf")


def test_upload_document_endpoint_invalid_file_type(client: TestClient) -> None:
    """Verify POST /api/v1/documents/upload rejects non-PDF files with 400 status."""
    file_content = b"This is a text file content."
    files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}

    response = client.post("/api/v1/documents/upload", files=files)

    assert response.status_code == 400
    assert "Only PDF documents are accepted" in response.json()["detail"]