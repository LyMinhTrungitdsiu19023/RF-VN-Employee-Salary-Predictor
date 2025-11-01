import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and encoders
def load_artifacts():
    model = joblib.load('salary_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    return model, scaler, label_encoders

model, scaler, label_encoders = load_artifacts()

# Feature options
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
job_roles = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'India']
genders = ['Male', 'Female', 'Other']

st.title('Employee Salary Prediction')
st.write('Predict employee salary using manual input or batch CSV upload.')
st.write('Developed by T Jagadish.')


# --- Manual Input ---
st.header('Manual Prediction')
with st.form('manual_form'):
    experience = st.slider('Years of Experience', 0, 20, 5)
    education = st.selectbox('Education Level', education_levels)
    job_role = st.selectbox('Job Role', job_roles)
    location = st.selectbox('Location', locations)
    age = st.slider('Age', 22, 60, 30)
    gender = st.selectbox('Gender', genders)
    submitted = st.form_submit_button('Predict Salary')

    if submitted:
        # Encode categorical features
        input_dict = {
            'YearsExperience': experience,
            'Education': label_encoders['Education'].transform([education])[0],
            'JobRole': label_encoders['JobRole'].transform([job_role])[0],
            'Location': label_encoders['Location'].transform([location])[0],
            'Age': age,
            'Gender': label_encoders['Gender'].transform([gender])[0],
        }
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        salary_pred = model.predict(input_scaled)[0]
        st.success(f'Predicted Salary: ₹{salary_pred:,.0f}')

# --- Batch Prediction ---
st.header('Batch Prediction (CSV Upload)')
st.write('Upload a CSV file with columns: YearsExperience, Education, JobRole, Location, Age, Gender')
file = st.file_uploader('Upload CSV', type=['csv'])
if file is not None:
    df_upload = pd.read_csv(file)
    # Check columns
    expected_cols = ['YearsExperience', 'Education', 'JobRole', 'Location', 'Age', 'Gender']
    if all(col in df_upload.columns for col in expected_cols):
        # Encode categorical columns
        for col in ['Education', 'JobRole', 'Location', 'Gender']:
            df_upload[col] = label_encoders[col].transform(df_upload[col])
        X_upload = df_upload[expected_cols]
        X_upload_scaled = scaler.transform(X_upload)
        salary_preds = model.predict(X_upload_scaled)
        df_upload['PredictedSalary'] = salary_preds.astype(int)
        st.dataframe(df_upload)
        # Download link
        csv = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button('Download Predictions as CSV', csv, 'salary_predictions.csv', 'text/csv')
    else:
        st.error(f'CSV must contain columns: {expected_cols}')

# --- (Optional) Model Metrics ---
st.header('Model & Dataset Info')
st.write('Trained on 500 synthetic employee records. Model: Random Forest Regressor.')
st.write('Features: YearsExperience, Education, JobRole, Location, Age, Gender')
st.write('All salaries are shown in Indian Rupees (₹).') 