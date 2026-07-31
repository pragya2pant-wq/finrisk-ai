"""
Pydantic Schemas for Credit Risk Scoring and SHAP Explainability.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ApplicantRiskInput(BaseModel):
    """Input financial features for tabular credit default prediction."""
    annual_income: float = Field(..., description="Annual income in currency", gt=0)
    debt_to_income_ratio: float = Field(..., description="DTI ratio as percentage (0-100)", ge=0, le=100)
    revolving_utilization: float = Field(..., description="Credit utilization ratio (0-1.0)", ge=0.0, le=1.0)
    num_delinquent_lines: int = Field(..., description="Number of past late payments", ge=0)
    num_credit_inquiries: int = Field(..., description="Hard credit inquiries in last 6 months", ge=0)
    loan_amount: float = Field(..., description="Requested loan amount", gt=0)
    credit_score: int = Field(..., description="FICO/CIBIL credit score (300-850)", ge=300, le=850)


class FeatureContribution(BaseModel):
    """Schema for individual SHAP feature impact score."""
    feature_name: str
    feature_value: float
    shap_value: float = Field(..., description="Positive increases default risk, negative decreases risk")
    impact_direction: str = Field(..., description="Increases Risk or Decreases Risk")


class CreditRiskPredictionOutput(BaseModel):
    """Output prediction payload returned by the credit ML inference engine."""
    applicant_id: Optional[str] = Field(default="APP-DEFAULT-001")
    probability_of_default: float = Field(..., description="Model predicted default probability (0.0 to 1.0)")
    credit_risk_grade: str = Field(..., description="Risk tier: Low Risk, Medium Risk, High Risk")
    approval_recommendation: str = Field(..., description="Automated credit decision recommendation")
    top_risk_drivers: List[FeatureContribution] = Field(..., description="Top SHAP feature contributions")