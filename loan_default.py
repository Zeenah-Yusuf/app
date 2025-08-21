import streamlit as st
import pandas as pd
import joblib

# --- Load model and encoders --
model = joblib.load("Model.pkl")
reason_encoder = joblib.load("Encoder.pkl")
job_encoder = joblib.load("Encoder.pkl")

st.set_page_config(page_title="Loan Default Risk Analyzer", page_icon="💰")

st.title("Loan Default Risk Analyzer")
st.write("Fill in the details below to estimate the risk of loan default.")

# --- User inputs ---
loan = st.number_input("Loan Amount", min_value=0.0, format="%.2f")
mortdue = st.number_input("Mortgage Due", min_value=0.0, format="%.2f")
value = st.number_input("Property Value", min_value=0.0, format="%.2f")
yoj = st.number_input("Years on Job", min_value=0.0, format="%.1f")
derog = st.number_input("Number of Derogatory Reports", min_value=0.0, format="%.0f")
delinq = st.number_input("Number of Delinquencies", min_value=0.0, format="%.0f")
clage = st.number_input("Age of Oldest Credit Line (months)", min_value=0.0, format="%.2f")
ninq = st.number_input("Recent Credit Inquiries", min_value=0.0, format="%.0f")
clno = st.number_input("Number of Credit Lines", min_value=0.0, format="%.0f")
debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0, format="%.2f")

reason_raw = st.selectbox("Loan Reason", reason_encoder.classes_)
job_raw = st.selectbox("Job Type", job_encoder.classes_)

# --- Predict button ---
if st.button("Predict Loan Risk"):
    try:
        # Encode categorical variables
        reason_encoded = reason_encoder.transform([reason_raw])[0]
        job_encoded = job_encoder.transform([job_raw])[0]

        # Prepare input in the same order as training
        input_data = pd.DataFrame([{
            'LOAN': loan,
            'MORTDUE': mortdue,
            'VALUE': value,
            'REASON': reason_encoded,
            'JOB': job_encoded,
            'YOJ': yoj,
            'DEROG': derog,
            'DELINQ': delinq,
            'CLAGE': clage,
            'NINQ': ninq,
            'CLNO': clno,
            'DEBTINC': debtinc
            
        }])

        # Make prediction
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][prediction]

        # Display result
        if prediction == 1:
            st.error(f" High risk of loan default ({proba*100:.1f}% probability)")
        else:
            st.success(f" Low risk of loan default ({proba*100:.1f}% probability)")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
