# FinRisk AI — Financial Risk Analytics & RAG System

A modular end-to-end system that combines Retrieval-Augmented Generation (RAG), ML Credit Scoring (XGBoost), and Financial Ratio Analysis into a single Streamlit dashboard.

---

## Features

1. **Document & RAG Search:**
   * Ingests financial PDFs into a ChromaDB vector store.
   * Performs semantic vector searches over extracted chunks using embeddings.

2. **Financial Ratio Engine:**
   * Calculates liquidity, solvency, and profitability metrics.
   * Assigns health status evaluations against industry benchmarks.

3. **Credit Risk ML & SHAP Explanations:**
   * Predicts default risk using an XGBoost model.
   * Explains feature contributions using SHAP values.

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** FastAPI, Pydantic, Uvicorn
* **ML / Vector DB:** XGBoost, SHAP, ChromaDB, SentenceTransformers
* **Language:** Python 3.10

---

## Setup & Run Instructions

1. **Activate Virtual Environment:**
   .\finrisk_env\Scripts\Activate.ps1

2. **Start FastAPI Backend (Terminal 1):**
   uvicorn app.main:app --reload

3. **Start Streamlit Frontend (Terminal 2):**
   streamlit run app.py
