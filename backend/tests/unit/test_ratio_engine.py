"""
Unit Tests for Financial Ratio Engine.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from app.schemas.financials import FinancialMetricsInput
from app.services.ratio_engine import FinancialRatioEngine


def test_ratio_engine_calculations() -> None:
    """
    Verify mathematical accuracy and health status evaluations for known financial input.
    """
    test_metrics = FinancialMetricsInput(
        current_assets=1500000.0,
        current_liabilities=1000000.0,
        inventory=200000.0,
        cash_and_equivalents=500000.0,
        total_debt=800000.0,
        total_equity=1000000.0,
        operating_income=300000.0,
        interest_expense=50000.0,
        net_income=200000.0,
        total_revenue=2000000.0
    )

    result = FinancialRatioEngine.calculate_all_ratios(test_metrics, "Test Co")

    # Current Ratio = 1,500,000 / 1,000,000 = 1.5
    assert result.liquidity_ratios["current_ratio"].value == 1.5
    assert result.liquidity_ratios["current_ratio"].health_status == "Healthy"

    # Debt to Equity = 800,000 / 1,000,000 = 0.8
    assert result.solvency_ratios["debt_to_equity"].value == 0.8
    assert result.solvency_ratios["debt_to_equity"].health_status == "Healthy"

    # Interest Coverage = 300,000 / 50,000 = 6.0
    assert result.solvency_ratios["interest_coverage"].value == 6.0

    # Overall score should be in healthy range (>70)
    assert result.overall_health_score > 70.0