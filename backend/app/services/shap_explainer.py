"""
SHAP Model Explainability Engine.

Calculates exact SHAP values for tree-based ensemble models (XGBoost)
to provide explainable AI features for credit decisions.

Author: Pragya Pant
Institute: iPEC Solutions
"""

from typing import List
import pandas as pd
import shap
import xgboost as xgb
from app.schemas.risk import FeatureContribution
from app.core.logging import logger


class SHAPExplainer:
    """
    Engine for generating SHAP feature impact scores for credit risk predictions.
    """

    def __init__(self, model: xgb.XGBClassifier) -> None:
        self.model = model
        # Initialize SHAP TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    def explain_prediction(self, feature_df: pd.DataFrame) -> List[FeatureContribution]:
        """
        Calculates SHAP values for a single applicant input vector.

        Returns:
            List[FeatureContribution]: Ranked list of top feature risk contributions.
        """
        logger.info("Calculating SHAP feature contributions for credit decision...")
        
        # Calculate raw SHAP matrix
        shap_values = self.explainer.shap_values(feature_df)

        # Handle 2D output array for binary classification
        if isinstance(shap_values, list):
            vals = shap_values[1][0]
        elif len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        feature_names = feature_df.columns.tolist()
        feature_vals = feature_df.iloc[0].values.tolist()

        contributions: List[FeatureContribution] = []
        for name, val, shap_val in zip(feature_names, feature_vals, vals):
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(val),
                    shap_value=round(float(shap_val), 4),
                    impact_direction="Increases Default Risk" if shap_val > 0 else "Decreases Default Risk"
                )
            )

        # Sort by absolute SHAP impact magnitude descending
        contributions.sort(key=lambda x: abs(x.shap_value), reverse=True)
        return contributions