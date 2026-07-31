"""
Unit tests for the Financial Ratio Calculation Engine.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import pytest
from app.schemas.financials import FinancialRatioInput
from app.services.ratio_engine import FinancialRatioEngine


def test_valid_ratio_calculation():
    """
    Test that valid financial inputs calculate expected liquidity and solvency ratios.
    """
    input_data = FinancialRatioInput(
        revenue=1000000.0,
        total_revenue=1000000.0,
        net_income=150000.0,
        operating_income=200000.0,
        current_assets=500000.0,
        current_liabilities=250000.0,
        inventory=50000.0,
        total_debt=300000.0,
        total_equity=600000.0,
        interest_expense=20000.0
    )

    result = FinancialRatioEngine.calculate_all_ratios(input_data)

    # Verify Current Ratio calculation (500000 / 250000 = 2.0)
    assert result.liquidity_ratios["current_ratio"].value == 2.0
    assert result.liquidity_ratios["current_ratio"].health_status == "Healthy"


def test_zero_liabilities_handling():
    """
    Test that zero current liabilities are handled safely without zero-division crashes.
    """
    input_data = FinancialRatioInput(
        revenue=500000.0,
        total_revenue=500000.0,
        net_income=50000.0,
        operating_income=70000.0,
        current_assets=200000.0,
        current_liabilities=0.0,
        inventory=10000.0,
        total_debt=100000.0,
        total_equity=300000.0,
        interest_expense=10000.0
    )

    result = FinancialRatioEngine.calculate_all_ratios(input_data)

    # Verify that zero division defaults gracefully
    assert result.liquidity_ratios["current_ratio"].value >= 0.0