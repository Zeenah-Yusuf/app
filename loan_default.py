
import streamlit as st
import pandas as pd
import joblib, traceback, streamlit as st

st.title("Loan Risk Analyzer")

try:
    model = joblib.load("Model.pkl")
except Exception:
    st.error("Failed to load Model.pkl – full traceback below:")
    st.text(traceback.format_exc())
    st.stop()


# Load model
model = joblib.load('Model.pkl')

# Set page config
st.set_page_config(page_title="Loan Risk Analyzer", layout="centered", page_icon="💼")

# Title
st.markdown("<h1 style='text-align: center; color: #00ffff;'>💼 Loan Default Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input fields
st.markdown("### Enter Loan Details")

loan = st.number_input("Loan Amount", min_value=0)
mortdue = st.number_input("Mortgage Due", min_value=0.0)
value = st.number_input("Property Value", min_value=0.0)
yoj = st.number_input("Years on Job", min_value=0)
derog = st.number_input("Number of Derogatory Reports", min_value=0)
delinq = st.number_input("Number of Delinquencies", min_value=0.0)
clage = st.number_input("Age of Oldest Credit Line", min_value=0.0)
ninq = st.number_input("Recent Credit Inquiries", min_value=0.0)
clno = st.number_input("Number of Credit Lines", min_value=0)
debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0)

reason = st.selectbox("Loan Reason", ["HomeImp", "DebtCon", "Missing"])
job = st.selectbox("Job Type", ["Other", "Office", "Sales", "Mgr", "ProfExe", "Self", "Missing"])

# Predict button
if st.button("Predict Loan Risk"):
    input_data = pd.DataFrame([{
        'LOAN': loan, 'MORTDUE': mortdue, 'VALUE': value, 'YOJ': yoj,
        'DEROG': derog, 'DELINQ': delinq, 'CLAGE': clage, 'NINQ': ninq,
        'CLNO': clno, 'DEBTINC': debtinc, 'REASON': reason, 'JOB': job
    }])

    prediction = model.predict(input_data)[0]
    result = "High Risk of Default" if prediction == 1 else "Low Risk of Default"

    st.markdown(f"<h2 style='text-align: center; color: {'red' if prediction == 1 else 'lime'};'>{result}</h2>", unsafe_allow_html=True)
