"""
Financial Ratio Calculation Engine.

Pure Python implementation for calculating Liquidity, Solvency, and Profitability ratios
with benchmark threshold analysis. Zero reliance on LLMs for math calculations.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from app.schemas.financials import (
    FinancialRatioInput,
    RatioCategoryResult,
    RatioAnalysisOutput
)
from app.core.logging import logger


class FinancialRatioEngine:
    """
    Deterministic calculation engine for core credit and financial risk metrics.
    """

    @staticmethod
    def calculate_all_ratios(
        metrics: FinancialRatioInput, company_name: str = "Target Entity"
    ) -> RatioAnalysisOutput:
        """
        Calculates all financial ratios and generates health status evaluations.

        Args:
            metrics (FinancialRatioInput): Validated raw financial input values.
            company_name (str): Name of company analyzed.

        Returns:
            RatioAnalysisOutput: Structured ratio analysis and overall score.
        """
        logger.info(f"Calculating financial ratios for company: {company_name}")

        # 1. Liquidity Calculations
        current_ratio = (
            metrics.current_assets / metrics.current_liabilities
            if metrics.current_liabilities > 0 else 0.0
        )
        quick_assets = metrics.current_assets - metrics.inventory
        quick_ratio = (
            quick_assets / metrics.current_liabilities
            if metrics.current_liabilities > 0 else 0.0
        )

        liquidity_dict = {
            "current_ratio": RatioCategoryResult(
                value=round(current_ratio, 2),
                health_status="Healthy" if current_ratio >= 1.5 else ("Warning" if current_ratio >= 1.0 else "Critical"),
                benchmark_note="Ideal benchmark >= 1.5x. Measures short-term obligation coverage."
            ),
            "quick_ratio": RatioCategoryResult(
                value=round(quick_ratio, 2),
                health_status="Healthy" if quick_ratio >= 1.0 else ("Warning" if quick_ratio >= 0.7 else "Critical"),
                benchmark_note="Ideal benchmark >= 1.0x. Excludes inventory from immediate liquidity."
            ),
            "working_capital": RatioCategoryResult(
                value=round(metrics.current_assets - metrics.current_liabilities, 2),
                health_status="Healthy" if (metrics.current_assets - metrics.current_liabilities) > 0 else "Critical",
                benchmark_note="Must be positive to ensure operational stability."
            )
        }

        # 2. Solvency / Leverage Calculations
        debt_to_equity = (
            metrics.total_debt / metrics.total_equity
            if metrics.total_equity != 0 else 0.0
        )
        interest_coverage = (
            metrics.operating_income / metrics.interest_expense
            if metrics.interest_expense > 0 else 999.0
        )

        solvency_dict = {
            "debt_to_equity": RatioCategoryResult(
                value=round(debt_to_equity, 2),
                health_status="Healthy" if debt_to_equity <= 1.5 else ("Warning" if debt_to_equity <= 2.5 else "Critical"),
                benchmark_note="Ideal benchmark <= 1.5x. Measures leverage risk."
            ),
            "interest_coverage": RatioCategoryResult(
                value=round(interest_coverage, 2),
                health_status="Healthy" if interest_coverage >= 3.0 else ("Warning" if interest_coverage >= 1.5 else "Critical"),
                benchmark_note="Ideal benchmark >= 3.0x. Measures ability to service debt interest."
            )
        }

        # 3. Profitability Calculations
        roe = (
            (metrics.net_income / metrics.total_equity) * 100
            if metrics.total_equity != 0 else 0.0
        )
        operating_margin = (
            (metrics.operating_income / metrics.total_revenue) * 100
            if metrics.total_revenue > 0 else 0.0
        )

        profitability_dict = {
            "return_on_equity_pct": RatioCategoryResult(
                value=round(roe, 2),
                health_status="Healthy" if roe >= 15.0 else ("Warning" if roe >= 8.0 else "Critical"),
                benchmark_note="Target benchmark >= 15.0%. Measures shareholder return efficiency."
            ),
            "operating_margin_pct": RatioCategoryResult(
                value=round(operating_margin, 2),
                health_status="Healthy" if operating_margin >= 12.0 else ("Warning" if operating_margin >= 5.0 else "Critical"),
                benchmark_note="Target benchmark >= 12.0%. Measures core operational profitability."
            )
        }

        # Composite Health Score Calculation (0 to 100)
        score = FinancialRatioEngine._calculate_composite_score(
            current_ratio, quick_ratio, debt_to_equity, interest_coverage, roe, operating_margin
        )

        return RatioAnalysisOutput(
            company_name=company_name,
            period="FY2025/2026",
            liquidity_ratios=liquidity_dict,
            solvency_ratios=solvency_dict,
            profitability_ratios=profitability_dict,
            overall_health_score=score
        )

    @staticmethod
    def _calculate_composite_score(
        cr: float, qr: float, de: float, ic: float, roe: float, om: float
    ) -> float:
        """Calculates a weighted overall credit health score out of 100."""
        score = 0.0
        # Liquidity (30 points)
        score += min(30.0, (cr / 1.5) * 15.0 + (qr / 1.0) * 15.0)
        # Solvency (35 points)
        score += min(20.0, (1.5 / max(de, 0.1)) * 20.0)
        score += min(15.0, (ic / 3.0) * 15.0)
        # Profitability (35 points)
        score += min(20.0, (roe / 15.0) * 20.0)
        score += min(15.0, (om / 12.0) * 15.0)

        return round(max(0.0, min(100.0, score)), 1)