"""
API Endpoints for Document Ingestion, PDF Parsing, and Vector Store Indexing.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import os
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.document import DocumentExtractionResponse
from app.schemas.rag import DocumentChunk
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.core.logging import logger

router = APIRouter()
vector_store = VectorStore()

UPLOAD_DIR = Path("./data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post(
    "/upload",
    response_model=DocumentExtractionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Parse Financial PDF Document"
)
async def upload_financial_document(file: UploadFile = File(...)) -> DocumentExtractionResponse:
    """
    Uploads a financial PDF document, stores it safely in raw storage,
    extracts text and tabular data, and indexes chunks into ChromaDB.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF documents are accepted."
        )

    file_path = UPLOAD_DIR / file.filename

    try:
        logger.info(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Extract text and tables
        extraction_result = PDFProcessor.extract_text_and_tables(str(file_path))

        # 2. Convert extracted text into a DocumentChunk for vector store
        if extraction_result.raw_text and extraction_result.raw_text.strip():
            chunk = DocumentChunk(
                chunk_id=f"chunk_{uuid.uuid4().hex[:8]}",
                document_name=file.filename,
                page_number=1,
                text_content=extraction_result.raw_text.strip()
            )
            # 3. Save directly to ChromaDB in the running server process
            vector_store.add_chunks([chunk])
            logger.info(f"Successfully indexed chunks for {file.filename} into vector store.")

        return extraction_result

    except Exception as e:
        logger.error(f"Failed to process document {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )