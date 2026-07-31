"""
Pydantic Schemas for Financial Metrics and Ratio Analysis.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class FinancialRatioInput(BaseModel):
    revenue: float = Field(..., ge=0, description="Revenue must be non-negative")
    total_revenue: float = Field(..., ge=0, description="Total revenue must be non-negative")
    net_income: float = Field(..., description="Net income can be positive or negative")
    operating_income: float = Field(..., description="Operating income can be positive or negative")
    current_assets: float = Field(..., ge=0, description="Current assets must be non-negative")
    current_liabilities: float = Field(..., ge=0, description="Current liabilities must be non-negative")
    total_debt: float = Field(..., ge=0, description="Total debt must be non-negative")
    total_equity: float = Field(..., description="Total equity")
    interest_expense: float = Field(..., ge=0, description="Interest expense must be non-negative")


class RatioCategoryResult(BaseModel):
    """Schema for individual financial ratio output."""
    value: float = Field(..., description="Calculated metric value")
    health_status: str = Field(..., description="Status evaluation: Healthy, Warning, Critical")
    benchmark_note: str = Field(..., description="Industry standard comparison note")


class RatioAnalysisOutput(BaseModel):
    """
    Comprehensive output schema containing calculated financial ratios categorized by financial domain.
    """
    company_name: Optional[str] = Field(default="Target Company", description="Name of evaluated entity")
    period: Optional[str] = Field(default="FY2025/2026", description="Reporting period")
    liquidity_ratios: Dict[str, RatioCategoryResult]
    solvency_ratios: Dict[str, RatioCategoryResult]
    profitability_ratios: Dict[str, RatioCategoryResult]
    overall_health_score: float = Field(..., description="Composite health score out of 100")