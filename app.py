import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
try:
    model = joblib.load('attrition_model.pkl')
    le_dept = joblib.load('dept_encoder.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run train_model.py first.")
    st.stop()

# Configure the Streamlit Page
st.set_page_config(page_title="DSPristine Retention Predictor", layout="centered")

# App Header
st.title("📊 DSPristine: Employee Retention Predictor")
st.markdown("Identify at-risk talent and proactively improve employee engagement.")

st.divider()

# User Input Form
st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=70, value=30)
    department = st.selectbox("Department", ["Sales", "R&D", "HR"])
    job_satisfaction = st.slider("Job Satisfaction Score", min_value=1, max_value=4, value=3, help="1 = Low, 4 = High")

with col2:
    years_at_company = st.number_input("Years at Company", min_value=0, max_value=50, value=5)
    monthly_income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=50000, value=5000, step=500)

# Prediction Logic
if st.button("Predict Retention Risk", type="primary"):
    
    # Encode the selected department
    encoded_dept = le_dept.transform([department])[0]
    
    # Create a DataFrame for the model
    input_data = pd.DataFrame({
        'Age': [age],
        'Department': [encoded_dept],
        'JobSatisfaction': [job_satisfaction],
        'YearsAtCompany': [years_at_company],
        'MonthlyIncome': [monthly_income]
    })
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    st.divider()
    
    # Display Results
    if prediction == 1:
        st.error(f"⚠️ **At Risk:** This employee is likely to LEAVE.")
        st.write(f"Confidence Level: **{probability[1] * 100:.1f}%**")
        st.info("Recommendation: Schedule a 1-on-1 meeting to discuss career growth or compensation.")
    else:
        st.success(f"✅ **Safe:** This employee is likely to STAY.")
        st.write(f"Confidence Level: **{probability[0] * 100:.1f}%**")