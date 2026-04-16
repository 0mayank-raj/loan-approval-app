import streamlit as st
import pickle
import pandas as pd

# -------------------- Setup --------------------
st.set_page_config(page_title="Loan Approval App", page_icon="💰", layout="wide")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# -------------------- Title --------------------
st.title("💰 Loan Approval Prediction")
st.write("Fill in the details below to check loan approval chances.")

st.divider()

# -------------------- Input Section --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal & Financial Info")

    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    age = st.number_input("Age", min_value=18, max_value=100)
    dependents = st.number_input("Dependents", min_value=0)
    savings = st.number_input("Savings", min_value=0)
    existing_loans = st.number_input("Existing Loans", min_value=0)

with col2:
    st.subheader("Loan & Profile Info")

    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    credit_score = st.slider("Credit Score", 300, 850, 650)
    dti_ratio = st.number_input("DTI Ratio", min_value=0.0)
    collateral_value = st.number_input("Collateral Value", min_value=0)

    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Education", "Personal"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    employer_category = st.selectbox("Employer Category", ["MNC", "Private", "Government", "Unemployed"])

st.divider()

# -------------------- Prepare Data --------------------
input_data = {
    "Applicant_Income": applicant_income,
    "Coapplicant_Income": coapplicant_income,
    "Loan_Amount": loan_amount,
    "Loan_Term": loan_term,
    "Credit_Score": credit_score,
    "Age": age,
    "Dependents": dependents,
    "DTI_Ratio": dti_ratio,
    "Savings": savings,
    "Existing_Loans": existing_loans,
    "Collateral_Value": collateral_value,
    "Employment_Status": employment_status,
    "Marital_Status": marital_status,
    "Education_Level": education_level,
    "Gender": gender,
    "Loan_Purpose": loan_purpose,
    "Property_Area": property_area,
    "Employer_Category": employer_category
}

input_df = pd.DataFrame([input_data])

# -------------------- Prediction --------------------
if st.button("Predict"):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if pred == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write(f"Approval Probability: {prob:.2f}")

    st.divider()

    # -------------------- Simple Insights --------------------
    st.subheader("Insights")

    if credit_score < 600:
        st.warning("Low credit score")

    if existing_loans > 2:
        st.warning("Too many existing loans")

    if dti_ratio > 0.4:
        st.warning("High debt-to-income ratio")

    if savings < loan_amount:
        st.warning("Savings are low compared to loan amount")

    if collateral_value < loan_amount:
        st.warning("Collateral is low")

    if credit_score >= 700 and existing_loans == 0:
        st.success("Strong financial profile")

# -------------------- Footer --------------------
st.divider()
st.caption("Built using Machine Learning and Streamlit")