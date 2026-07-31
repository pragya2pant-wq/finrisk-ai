"""
XGBoost ML Credit Risk Scoring Engine.

Handles training on tabular credit data, saving serialized model artifacts,
and serving real-time Probability of Default (PD) predictions.

Author: Pragya Pant
Institute: iPEC Solutions
"""

import os
from pathlib import Path
from typing import Tuple, Optional  
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from app.schemas.risk import ApplicantRiskInput
from app.core.logging import logger

MODEL_DIR = Path("./data/models")
MODEL_PATH = MODEL_DIR / "credit_xgboost_v1.pkl"


class CreditMLEngine:
    """
    Service for XGBoost credit risk scoring model training and inference.
    """

    def __init__(self) -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.model: Optional[xgb.XGBClassifier] = None
        self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Loads serialized model from disk or trains a new synthetic baseline if missing."""
        if MODEL_PATH.exists():
            logger.info(f"Loading existing XGBoost credit model from {MODEL_PATH}")
            self.model = joblib.load(MODEL_PATH)
        else:
            logger.info("No pre-trained model found. Training synthetic XGBoost credit model...")
            self.train_synthetic_model()

    def train_synthetic_model(self) -> None:
        """Trains a domain-accurate synthetic XGBoost model for credit risk evaluation."""
        np.random.seed(42)
        n_samples = 2000

        # Generate synthetic credit dataset
        annual_income = np.random.uniform(20000, 200000, n_samples)
        dti = np.random.uniform(5, 60, n_samples)
        revolving_util = np.random.uniform(0.05, 0.95, n_samples)
        delinquencies = np.random.poisson(0.5, n_samples)
        inquiries = np.random.poisson(1.0, n_samples)
        loan_amount = np.random.uniform(5000, 100000, n_samples)
        credit_score = np.random.randint(550, 850, n_samples)

        df = pd.DataFrame({
            "annual_income": annual_income,
            "debt_to_income_ratio": dti,
            "revolving_utilization": revolving_util,
            "num_delinquent_lines": delinquencies,
            "num_credit_inquiries": inquiries,
            "loan_amount": loan_amount,
            "credit_score": credit_score
        })

        # Define ground truth default probability formula
        logit = (
            -2.5 
            + (dti * 0.04) 
            + (revolving_util * 2.5) 
            + (delinquencies * 0.6) 
            + (inquiries * 0.3) 
            - (annual_income / 50000) 
            - (credit_score - 600) * 0.01
        )
        prob = 1 / (1 + np.exp(-logit))
        y = (prob > 0.5).astype(int)

        # Train XGBoost Classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
        self.model.fit(df, y)

        # Save model artifact
        joblib.dump(self.model, MODEL_PATH)
        logger.info(f"Model successfully trained and saved to {MODEL_PATH}")

    def predict_risk(self, applicant_data: ApplicantRiskInput) -> Tuple[float, str, str, pd.DataFrame]:
        """
        Executes ML inference on applicant feature vector.

        Returns:
            Tuple: (Probability of Default, Risk Grade, Recommendation, Feature DataFrame)
        """
        if self.model is None:
            self._load_or_train_model()

        input_df = pd.DataFrame([applicant_data.model_dump()])

        # Predict Probability of Default (Class 1 = Default)
        prob_default = float(self.model.predict_proba(input_df)[0][1])

        # Assign Risk Tiers based on banking industry thresholds
        if prob_default < 0.20:
            grade = "Low Risk"
            recommendation = "Approved - Standard Terms"
        elif prob_default < 0.45:
            grade = "Medium Risk"
            recommendation = "Conditional Approval - Requires Higher Collateral"
        else:
            grade = "High Risk"
            recommendation = "Rejected - High Probability of Default"

        return prob_default, grade, recommendation, input_df