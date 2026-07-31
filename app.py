import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000/api/v1"

st.set_page_config(page_title="FinRisk AI Dashboard", layout="wide")
st.title("FinRisk AI Platform")

tab1, tab2, tab3 = st.tabs(["Document & RAG Search", "Financial Ratios", "Credit Risk ML"])

# Tab 1: Document Upload and RAG Search
with tab1:
    st.header("Document Ingestion & Semantic Search")
    uploaded_file = st.file_uploader("Upload Financial PDF", type=["pdf"])
    if uploaded_file and st.button("Upload & Index"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        res = requests.post(f"{API_BASE_URL}/documents/upload", files=files)
        if res.status_code == 201:
            st.success("Document indexed successfully into ChromaDB!")
        else:
            st.error(f"Error: {res.text}")

    st.subheader("Semantic Search")
    query = st.text_input("Enter search query:")
    if query and st.button("Search"):
        res = requests.post(f"{API_BASE_URL}/rag/search", json={"query": query, "top_k": 5})
        if res.status_code == 200:
            results = res.json().get("retrieved_chunks", [])
            
            # Deduplicate results based on exact text content
            seen_texts = set()
            unique_results = []
            for chunk in results:
                if chunk['text_content'] not in seen_texts:
                    seen_texts.add(chunk['text_content'])
                    unique_results.append(chunk)

            for chunk in unique_results:
                st.write(f"**Document:** {chunk['document_name']} (Page {chunk['page_number']})")
                st.write(f"**Similarity Score:** {chunk['similarity_score']}")
                st.info(chunk['text_content'])
        else:
            st.error(f"Error: {res.text}")

# Tab 2: Financial Ratio Calculator
with tab2:
    st.header("Financial Ratio Engine")
    col1, col2 = st.columns(2)
    with col1:
        revenue = st.number_input("Revenue", value=100000.0)
        total_revenue = st.number_input("Total Revenue", value=100000.0)
        net_income = st.number_input("Net Income", value=15000.0)
        operating_income = st.number_input("Operating Income", value=20000.0)
        current_assets = st.number_input("Current Assets", value=50000.0)
    with col2:
        current_liabilities = st.number_input("Current Liabilities", value=25000.0)
        total_debt = st.number_input("Total Debt", value=40000.0)
        total_equity = st.number_input("Total Equity", value=60000.0)
        interest_expense = st.number_input("Interest Expense", value=2000.0)

    if st.button("Calculate Ratios"):
        payload = {
            "revenue": revenue,
            "total_revenue": total_revenue,
            "net_income": net_income,
            "operating_income": operating_income,
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "total_debt": total_debt,
            "total_equity": total_equity,
            "interest_expense": interest_expense
        }
        res = requests.post(f"{API_BASE_URL}/financials/ratios", json=payload)
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error(f"Error: {res.text}")

# Tab 3: Credit Risk ML Assessment
with tab3:
    st.header("Credit Risk ML Scoring & SHAP Explanations")
    col1, col2 = st.columns(2)
    with col1:
        annual_income = st.number_input("Annual Income", value=85000.0)
        credit_score = st.number_input("Credit Score", value=720)
        loan_amount = st.number_input("Loan Amount", value=15000.0)
        debt_to_income_ratio = st.number_input("Debt to Income Ratio", value=0.25)
    with col2:
        revolving_utilization = st.number_input("Revolving Utilization", value=0.30)
        num_delinquent_lines = st.number_input("Number of Delinquent Lines", value=0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", value=1)

    if st.button("Predict Credit Risk"):
        payload = {
            "annual_income": annual_income,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "debt_to_income_ratio": debt_to_income_ratio,
            "revolving_utilization": revolving_utilization,
            "num_delinquent_lines": num_delinquent_lines,
            "num_credit_inquiries": num_credit_inquiries
        }
        res = requests.post(f"{API_BASE_URL}/risk/predict", json=payload)
        if res.status_code == 200:
            st.json(res.json())
        else:
            st.error(f"Error: {res.text}")
