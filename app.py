import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="CreditWise", page_icon="💳", layout="centered")

# ---------- Modern minimal styling ----------
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }
    .block-container {
        padding-top: 3rem;
        max-width: 620px;
    }
    h1 {
        font-weight: 700;
        color: #f8fafc;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-size: 0.95rem;
    }
    div[data-testid="stForm"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
    }
    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        border: 1px solid #334155 !important;
    }
    .stButton button, .stFormSubmitButton button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem;
        font-weight: 600;
        font-size: 1rem;
    }
    .result-card {
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .approved { background: rgba(34,197,94,0.15); border: 1px solid #22c55e; color: #4ade80; }
    .rejected { background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #f87171; }
</style>
""", unsafe_allow_html=True)

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
defaults = artifacts.get("defaults", {
    "Coapplicant_Income": 1200.0,
    "Age": 35,
    "Dependents": 0,
    "Existing_Loans": 0,
    "Savings": 5000.0,
    "Collateral_Value": 10000.0,
    "Marital_Status": "Married",
    "Property_Area": "Urban",
    "Gender": "Male",
    "Employer_Category": "Private",
    "Education_Level": "Graduate",
})

st.markdown("<h1>💳 CreditWise</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Quick loan approval check</p>', unsafe_allow_html=True)

with st.form("loan_form"):
    applicant_income = st.number_input("Monthly Income", min_value=0.0, value=8000.0, step=500.0)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=650)
    dti_ratio = st.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    loan_amount = st.number_input("Loan Amount Requested", min_value=0.0, value=10000.0, step=500.0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 84], index=2)
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Personal", "Business"])

    submitted = st.form_submit_button("Check Approval")

if submitted:
    raw = pd.DataFrame([{
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": defaults["Coapplicant_Income"],
        "Employment_Status": employment_status,
        "Age": defaults["Age"],
        "Marital_Status": defaults["Marital_Status"],
        "Dependents": defaults["Dependents"],
        "Credit_Score": credit_score,
        "Existing_Loans": defaults["Existing_Loans"],
        "DTI_Ratio": dti_ratio,
        "Savings": defaults["Savings"],
        "Collateral_Value": defaults["Collateral_Value"],
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Loan_Purpose": loan_purpose,
        "Property_Area": defaults["Property_Area"],
        "Education_Level": defaults["Education_Level"],
        "Gender": defaults["Gender"],
        "Employer_Category": defaults["Employer_Category"],
    }])

    raw["Education_Level"] = le_education.transform(raw["Education_Level"])
    encoded = ohe.transform(raw[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=raw.index)
    raw = pd.concat([raw.drop(columns=ohe_cols), encoded_df], axis=1)

    raw["DTI_Ratio_sq"] = raw["DTI_Ratio"] ** 2
    raw["Credit_Score_sq"] = raw["Credit_Score"] ** 2
    raw = raw.drop(columns=["Credit_Score", "DTI_Ratio"])
    raw = raw.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(raw)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    label = le_target.inverse_transform([pred])[0]
    approved_idx = list(le_target.classes_).index("Yes") if "Yes" in le_target.classes_ else pred

    if label == "Yes":
        st.markdown(
            f'<div class="result-card approved">✅ Loan Likely Approved<br>'
            f'<span style="font-size:0.9rem; font-weight:400;">Confidence: {proba[approved_idx]*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card rejected">❌ Loan Likely Not Approved<br>'
            f'<span style="font-size:0.9rem; font-weight:400;">Confidence: {(1-proba[approved_idx])*100:.1f}%</span></div>',
            unsafe_allow_html=True,
        )
