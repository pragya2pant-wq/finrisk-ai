"""
Pydantic Schemas for Document Parsing and Metadata Processing.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ExtractedTable(BaseModel):
    """Schema for extracted financial table from PDF."""
    page_number: int
    headers: List[str]
    rows: List[List[str]]


class DocumentExtractionResponse(BaseModel):
    """Schema for completed document processing output."""
    filename: str
    total_pages: int
    raw_text: str = Field(..., description="Cleaned aggregated text from PDF")
    extracted_tables: List[ExtractedTable]
    metadata: Dict[str, Any]