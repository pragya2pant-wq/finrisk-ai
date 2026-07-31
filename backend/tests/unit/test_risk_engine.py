"""
Unit Tests for XGBoost Credit ML Engine and SHAP Explainer.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from app.schemas.risk import ApplicantRiskInput
from app.services.credit_ml_engine import CreditMLEngine
from app.services.shap_explainer import SHAPExplainer


def test_credit_ml_engine_prediction() -> None:
    """Verify that ML credit scoring returns valid probability bounds and risk grade."""
    engine = CreditMLEngine()
    
    # High risk applicant profile
    high_risk_applicant = ApplicantRiskInput(
        annual_income=25000.0,
        debt_to_income_ratio=55.0,
        revolving_utilization=0.90,
        num_delinquent_lines=3,
        num_credit_inquiries=4,
        loan_amount=50000.0,
        credit_score=580
    )

    prob_default, grade, rec, df = engine.predict_risk(high_risk_applicant)

    # Valid probability range
    assert 0.0 <= prob_default <= 1.0
    # Expected High Risk grade for bad applicant
    assert grade in ["Medium Risk", "High Risk"]
    
    # Verify SHAP explanations
    explainer = SHAPExplainer(engine.model)
    contributions = explainer.explain_prediction(df)
    assert len(contributions) == 7
    assert contributions[0].shap_value != 0.0