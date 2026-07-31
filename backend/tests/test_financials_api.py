"""
Unit tests for Financial Analysis API Endpoints.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import pytest
from fastapi.testclient import TestClient


def test_calculate_financial_ratios_endpoint_success(client: TestClient):
    """
    Test POST /api/v1/financials/ratios with valid payload returns 200 OK and valid schema.
    """
    payload = {
        "revenue": 1000000.0,
        "total_revenue": 1000000.0,
        "net_income": 150000.0,
        "operating_income": 200000.0,
        "current_assets": 500000.0,
        "current_liabilities": 250000.0,
        "inventory": 50000.0,
        "total_debt": 300000.0,
        "total_equity": 600000.0,
        "interest_expense": 20000.0
    }

    response = client.post("/api/v1/financials/ratios?company_name=Acme%20Corp", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["company_name"] == "Acme Corp"
    assert "liquidity_ratios" in data
    assert "solvency_ratios" in data
    assert "profitability_ratios" in data
    assert data["liquidity_ratios"]["current_ratio"]["value"] == 2.0


def test_calculate_financial_ratios_endpoint_validation_error(client: TestClient):
    """
    Test POST /api/v1/financials/ratios with missing required fields returns 422 Unprocessable Entity.
    """
    invalid_payload = {
        "revenue": 1000000.0
        # Missing required financial fields like current_assets, total_debt, etc.
    }

    response = client.post("/api/v1/financials/ratios", json=invalid_payload)

    assert response.status_code == 422