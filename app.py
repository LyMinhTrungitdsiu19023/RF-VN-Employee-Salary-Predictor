import streamlit as st
import numpy as np
import joblib

# --- Load model, scaler, and encoders ---
model = joblib.load('salary_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# --- App UI ---
st.set_page_config(page_title="Employee Salary Predictor", page_icon="💼", layout="centered", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("💼 Employee Salary Predictor")
st.sidebar.markdown("""
**Predict the salary of an employee based on their profile!**

*Powered by Machine Learning*  
*Developed by [T Jagadish]*
""")

# Main header with custom style
st.markdown("""
    <style>
    .main-title {
        font-size:2.5em;
        font-weight:700;
        color:#2E86C1;
        text-align:center;
        margin-bottom:0.2em;
    }
    .subtitle {
        font-size:1.2em;
        color:#117A65;
        text-align:center;
        margin-bottom:2em;
    }
    </style>
    <div class="main-title">Employee Salary Prediction App 💰</div>
    <div class="subtitle">Enter employee details to estimate their annual salary in INR</div>
""", unsafe_allow_html=True)

# --- Input widgets ---
col1, col2 = st.columns(2)

with col1:
    experience = st.slider("Years of Experience", 0, 20, 2)
    age = st.slider("Age", 22, 60, 30)
    education = st.selectbox("Education Level", label_encoders['Education'].classes_)
    job_role = st.selectbox("Job Role", label_encoders['JobRole'].classes_)

with col2:
    location = st.selectbox("Location", label_encoders['Location'].classes_)
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)

# --- Prediction logic ---
def predict_salary(experience, education, job_role, location, age, gender):
    # Encode categorical features
    education_enc = label_encoders['Education'].transform([education])[0]
    job_role_enc = label_encoders['JobRole'].transform([job_role])[0]
    location_enc = label_encoders['Location'].transform([location])[0]
    gender_enc = label_encoders['Gender'].transform([gender])[0]
    # Prepare feature array
    features = np.array([[experience, education_enc, job_role_enc, location_enc, age, gender_enc]])
    features_scaled = scaler.transform(features)
    salary_pred = model.predict(features_scaled)[0]
    return int(salary_pred)

if st.button("Predict Salary", use_container_width=True):
    salary = predict_salary(experience, education, job_role, location, age, gender)
    st.success(f"Estimated Annual Salary: ₹ {salary:,.0f}")
    st.markdown(f"<div style='text-align:center; font-size:1.5em; color:#D35400;'>🎉 This employee can expect a salary of <b>₹ {salary:,.0f}</b> per year!</div>", unsafe_allow_html=True)
    st.progress(min(salary/4000000, 1.0))

# --- Optional: Data exploration ---
with st.expander("Show Example Data Insights"):
    import pandas as pd
    df = pd.read_csv('employee_salary_data.csv')
    st.write(df.sample(5))
    st.bar_chart(df['Salary']) 