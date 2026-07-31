"""
Pydantic Schemas for Financial Metrics and Ratio Analysis.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class FinancialMetricsInput(BaseModel):
    """
    Input schema containing raw financial statement values required for ratio calculations.
    Values are expected in standard monetary currency (e.g., INR or USD).
    """
    current_assets: float = Field(..., description="Total current assets", gt=0)
    current_liabilities: float = Field(..., description="Total current liabilities", gt=0)
    inventory: float = Field(default=0.0, description="Total inventory value", ge=0)
    cash_and_equivalents: float = Field(default=0.0, description="Cash and liquid equivalents", ge=0)
    total_debt: float = Field(..., description="Total debt (Short-term + Long-term debt)", ge=0)
    total_equity: float = Field(..., description="Total shareholders equity", gt=0)
    operating_income: float = Field(..., description="Operating Income / EBIT", ge=0)
    interest_expense: float = Field(..., description="Total annual interest expense", ge=0)
    net_income: float = Field(..., description="Net Income after tax")
    total_revenue: float = Field(..., description="Total annual revenue", gt=0)


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