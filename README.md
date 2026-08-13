# FinRisk AI — Financial Risk Analytics Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finrisk-ai-gldkkxtgauhvsoqs4obafq.streamlit.app/)

**Live Application:** [finrisk-ai.streamlit.app](https://finrisk-ai-gldkkxtgauhvsoqs4obafq.streamlit.app/)

An enterprise-grade financial analytics system pairing a FastAPI REST API backend with a Streamlit executive dashboard.

---

## 👤 Author Information

* **Developer:** Pragya Pant
* **LinkedIn:** [Pragya Pant](https://www.linkedin.com/in/pragyapant4827/)
* **GitHub:** [pragya2pant-wq](https://github.com/pragya2pant-wq)
* **Role:** AI/ML Developer — Finance Applications
* **Institute:** iPEC Solutions Pvt. Ltd, Bangalore
* **Built:** August 2026
* **Version:** v1.0.0

> ⚠️ **Disclaimer:** This is a portfolio/demonstration project built using synthetic/public datasets for educational purposes. It is not intended for actual credit or lending decisions.

---

## Key Features

1. **Document Ingestion & RAG Executive Search:**
   * Ingests financial PDFs into a ChromaDB vector store.
   * Uses semantic search and cosine similarity matching to return reference chunks.

2. **Financial Ratio Engine:**
   * Computes key liquidity, solvency, and profitability metrics.
   * Evaluates metrics against financial benchmarks to derive an overall health score.

3. **Credit Risk ML Scoring & SHAP Explainability:**
   * Predicts Probability of Default (PD) and risk grades using XGBoost.
   * Analyzes local feature contributions for predictions using SHAP values.

4. **Model Benchmarks & Evaluation:**
   * Displays model performance metrics (ROC-AUC, Precision, F1-Score, Recall).

5. **Prediction Audit Log & Compliance History:**
   * Tracks session inferences and allows exporting log records as CSV files.

---

## Tech Stack

* **Backend:** FastAPI, Pydantic v2, Uvicorn
* **Frontend:** Streamlit, Plotly Express, Requests
* **Machine Learning & Analytics:** XGBoost, SHAP, NumPy, Pandas, Scikit-Learn
* **Vector Store & Embeddings:** ChromaDB, Sentence-Transformers, PyPDF

---

## Local Execution

### 1. Start the FastAPI Backend (Terminal 1)
```bash
uvicorn backend.app.main:app --reload --port 8000