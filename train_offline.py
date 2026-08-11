import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Generate 100,000-row dataset with exact UI feature names
n = 100000
np.random.seed(42)

df = pd.DataFrame({
    "annual_income": np.random.uniform(20000, 200000, n),
    "debt_to_income_ratio": np.random.uniform(5, 60, n),
    "revolving_utilization": np.random.uniform(0.05, 0.95, n),
    "num_delinquent_lines": np.random.poisson(0.5, n),
    "num_credit_inquiries": np.random.poisson(1.0, n),
    "loan_amount": np.random.uniform(5000, 100000, n),
    "credit_score": np.random.randint(550, 850, n),
})

# Ground truth probability equation
logit = (
    -2.5
    + (df["debt_to_income_ratio"] * 0.04)
    + (df["revolving_utilization"] * 2.5)
    + (df["num_delinquent_lines"] * 0.6)
    + (df["num_credit_inquiries"] * 0.3)
    - (df["annual_income"] / 50000)
    - (df["credit_score"] - 600) * 0.01
)
prob = 1 / (1 + np.exp(-logit))
df["TARGET"] = (prob > 0.5).astype(int)

feature_columns = [
    "annual_income",
    "debt_to_income_ratio",
    "revolving_utilization",
    "num_delinquent_lines",
    "num_credit_inquiries",
    "loan_amount",
    "credit_score",
]

X = df[feature_columns]
y = df["TARGET"]

# 2. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train XGBoost Model
print("Training XGBoost model on 100,000 rows with exact UI names...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# 4. Save Model Artifact
output_model_path = "backend/data/models/credit_xgboost_v1.pkl"
joblib.dump(model, output_model_path)
print(f"✅ Success! Trained model artifact saved to: {output_model_path}")