import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="CreditWise - Loan Approval Predictor", page_icon="💳", layout="centered")

@st.cache_resource
def load_artifacts():
    return joblib.load("credit_model.pkl")

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
ohe = artifacts["ohe"]
le_education = artifacts["le_education"]
le_target = artifacts["le_target"]
ohe_cols = artifacts["ohe_cols"]
feature_columns = artifacts["feature_columns"]

st.title("💳 CreditWise")
st.write("Loan approval predictor — Naive Bayes model")

with st.form("loan_form"):
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)
    with col1:
        applicant_income = st.number_input("Applicant Income", min_value=0.0, value=8000.0, step=100.0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=1000.0, step=100.0)
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0)
    with col2:
        dti_ratio = st.slider("DTI Ratio (Debt-to-Income)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        savings = st.number_input("Savings", min_value=0.0, value=5000.0, step=100.0)
        collateral_value = st.number_input("Collateral Value", min_value=0.0, value=10000.0, step=100.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=10000.0, step=100.0)
        loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)

    st.subheader("Categorical Details")
    col3, col4 = st.columns(2)
    with col3:
        employment_status = st.selectbox("Employment Status", ["Contract", "Salaried", "Self-employed", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        loan_purpose = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
        gender = st.selectbox("Gender", ["Female", "Male"])
    with col4:
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
        employer_category = st.selectbox("Employer Category", ["Business", "Government", "MNC", "Private", "Unemployed"])
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    submitted = st.form_submit_button("Predict Loan Approval")

if submitted:
    # Build a single-row dataframe matching the raw training columns
    raw = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Employment_Status": employment_status,
        "Age": age,
        "Marital_Status": marital_status,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": property_area,
        "Education_Level": education_level,
        "Gender": gender,
        "Employer_Category": employer_category,
    }])

    # Same encoding steps as training
    raw["Education_Level"] = le_education.transform(raw["Education_Level"])

    encoded = ohe.transform(raw[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=raw.index)
    raw = pd.concat([raw.drop(columns=ohe_cols), encoded_df], axis=1)

    # Same feature engineering as training
    raw["DTI_Ratio_sq"] = raw["DTI_Ratio"] ** 2
    raw["Credit_Score_sq"] = raw["Credit_Score"] ** 2
    raw = raw.drop(columns=["Credit_Score", "DTI_Ratio"])

    # Ensure exact column order the model was trained on
    raw = raw.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(raw)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]

    label = le_target.inverse_transform([pred])[0]
    approved_idx = list(le_target.classes_).index("Yes") if "Yes" in le_target.classes_ else pred

    st.divider()
    if label == "Yes":
        st.success(f"✅ Loan Approved (confidence: {proba[approved_idx]*100:.1f}%)")
    else:
        st.error(f"❌ Loan Not Approved (confidence: {(1-proba[approved_idx])*100:.1f}%)")

    with st.expander("See prediction probabilities"):
        prob_df = pd.DataFrame({"Outcome": le_target.classes_, "Probability": proba})
        st.bar_chart(prob_df.set_index("Outcome"))
