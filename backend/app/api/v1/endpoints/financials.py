"""
API Endpoints for Financial Ratio Calculation and Credit Analysis.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.financials import (
    FinancialRatioInput,
    RatioAnalysisOutput
)
from app.services.ratio_engine import FinancialRatioEngine
from app.core.logging import logger

router = APIRouter()


@router.post(
    "/ratios",
    response_model=RatioAnalysisOutput,
    status_code=status.HTTP_200_OK,
    summary="Calculate Financial Ratios & Credit Health Score"
)
async def calculate_financial_ratios(
    metrics: FinancialRatioInput, company_name: str = "Target Entity"
) -> RatioAnalysisOutput:
    """
    Executes pure-Python deterministic financial ratio calculations, health benchmarks,
    and returns a composite credit health score.
    """
    try:
        results = FinancialRatioEngine.calculate_all_ratios(metrics, company_name)
        return results
    except Exception as e:
        logger.error(f"Error calculating ratios: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Financial ratio calculation failed: {str(e)}"
        )