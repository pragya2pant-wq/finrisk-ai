"""
Unit Tests for XGBoost Credit ML Engine, SHAP Explainer, and Risk Scoring API Endpoint.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from fastapi.testclient import TestClient
from app.schemas.risk import ApplicantRiskInput
from app.services.credit_ml_engine import CreditMLEngine
from app.services.shap_explainer import SHAPExplainer


def test_credit_ml_engine_prediction_high_risk() -> None:
    """Verify that ML credit scoring identifies high-risk profiles accurately."""
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

    # Valid probability range and high risk grade assertion
    assert 0.0 <= prob_default <= 1.0
    assert grade in ["Medium Risk", "High Risk"]
    
    # Verify SHAP explanations
    explainer = SHAPExplainer(engine.model)
    contributions = explainer.explain_prediction(df)
    assert len(contributions) == 7


def test_credit_ml_engine_prediction_low_risk() -> None:
    """Verify that ML credit scoring identifies low-risk profiles accurately."""
    engine = CreditMLEngine()
    
    # Low risk applicant profile
    low_risk_applicant = ApplicantRiskInput(
        annual_income=150000.0,
        debt_to_income_ratio=10.0,
        revolving_utilization=0.10,
        num_delinquent_lines=0,
        num_credit_inquiries=0,
        loan_amount=10000.0,
        credit_score=800
    )

    prob_default, grade, rec, df = engine.predict_risk(low_risk_applicant)

    # Valid probability range and low risk grade assertion
    assert 0.0 <= prob_default <= 1.0
    assert grade in ["Low Risk", "Medium Risk"]


def test_predict_credit_risk_endpoint_success(client: TestClient) -> None:
    """Verify POST /api/v1/risk/predict returns 200 OK with prediction output schema."""
    payload = {
        "annual_income": 75000.0,
        "debt_to_income_ratio": 25.0,
        "revolving_utilization": 0.35,
        "num_delinquent_lines": 0,
        "num_credit_inquiries": 1,
        "loan_amount": 20000.0,
        "credit_score": 710
    }

    response = client.post("/api/v1/risk/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "applicant_id" in data
    assert "probability_of_default" in data
    assert "credit_risk_grade" in data
    assert "approval_recommendation" in data
    assert "top_risk_drivers" in data
    assert len(data["top_risk_drivers"]) == 7