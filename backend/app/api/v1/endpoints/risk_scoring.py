"""
API Endpoints for Machine Learning Credit Scoring and SHAP Analysis.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas.risk import ApplicantRiskInput, CreditRiskPredictionOutput
from app.services.credit_ml_engine import CreditMLEngine
from app.services.shap_explainer import SHAPExplainer
from app.core.logging import logger

router = APIRouter()

# Instantiate ML Service Engine
ml_engine = CreditMLEngine()


@router.post(
    "/predict",
    response_model=CreditRiskPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Predict Credit Default Risk with SHAP Explainability"
)
async def predict_credit_risk(applicant: ApplicantRiskInput) -> CreditRiskPredictionOutput:
    """
    Runs XGBoost ML credit scoring model to predict Probability of Default (PD),
    assigns a risk grade, and generates top SHAP feature drivers.
    """
    try:
        prob_default, grade, recommendation, input_df = ml_engine.predict_risk(applicant)

        # Generate SHAP explanations
        explainer = SHAPExplainer(ml_engine.model)
        shap_contributions = explainer.explain_prediction(input_df)

        return CreditRiskPredictionOutput(
            applicant_id="APP-2026-X89",
            probability_of_default=round(prob_default, 4),
            credit_risk_grade=grade,
            approval_recommendation=recommendation,
            top_risk_drivers=shap_contributions
        )

    except Exception as e:
        logger.error(f"Error executing ML risk scoring: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Credit ML scoring failed: {str(e)}"
        )