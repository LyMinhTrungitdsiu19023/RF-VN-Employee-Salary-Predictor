import streamlit as st
import time
import os
import pandas as pd
import numpy as np
import joblib

# Predefined user accounts
USERS = {
    # Admin accounts
    "admin1": {"password": "admin123", "role": "Admin"},
    "admin2": {"password": "admin456", "role": "Admin"},
    "admin3": {"password": "admin789", "role": "Admin"},
    
    # User accounts - Employers
    "employer1": {"password": "emp123", "role": "User", "user_type": "Employer"},
    "employer2": {"password": "emp456", "role": "User", "user_type": "Employer"},
    
    # User accounts - Job Seekers
    "seeker1": {"password": "seek123", "role": "User", "user_type": "Job Seeker"},
}

def main():
    st.title("Employee System")
    
    # Initialize session state variables if they don't exist
    if 'page' not in st.session_state:
        st.session_state.page = 'role_selection'
    if 'role' not in st.session_state:
        st.session_state.role = None
    
    # Display different pages based on session state
    if st.session_state.page == 'role_selection':
        display_role_selection()
    elif st.session_state.page == 'login':
        display_login_page()
    elif st.session_state.page == 'dashboard':
        display_dashboard()
    elif st.session_state.page == 'salary_prediction':
        display_salary_prediction()

def display_role_selection():
    st.header("Select Your Role")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Admin", use_container_width=True):
            st.session_state.role = "Admin"
            st.session_state.page = 'login'
            st.rerun()
    
    with col2:
        if st.button("User", use_container_width=True):
            st.session_state.role = "User"
            st.session_state.page = 'login'
            st.rerun()
    
    st.markdown("---")
    st.write("Please select a role to continue to the login page.")
    
    # Display available test accounts
    with st.expander("Available Test Accounts"):
        st.markdown("### Admin Accounts")
        st.markdown("- Username: `admin1`, Password: `admin123`")
        st.markdown("- Username: `admin2`, Password: `admin456`")
        st.markdown("- Username: `admin3`, Password: `admin789`")
        
        st.markdown("### Employer Accounts")
        st.markdown("- Username: `employer1`, Password: `emp123`")
        st.markdown("- Username: `employer2`, Password: `emp456`")
        
        st.markdown("### Job Seeker Accounts")
        st.markdown("- Username: `seeker1`, Password: `seek123`")

def display_login_page():
    st.header(f"Login as {st.session_state.role}")
    
    # Back button
    if st.button("← Back to Role Selection"):
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.rerun()
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        # Only show user type selection for regular users if not predefined
        user_type = None
        if st.session_state.role == "User":
            # Check if username exists and has a predefined user_type
            if username in USERS and USERS[username]["role"] == "User":
                user_type = USERS[username]["user_type"]
                st.info(f"Account type: {user_type}")
            else:
                user_type = st.selectbox("User Type", ["Employer", "Job Seeker"])
        
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            login_result = validate_login(username, password, st.session_state.role, user_type)
            if login_result["success"]:
                st.success("Login successful!")
                # Store user info in session state
                st.session_state.username = username
                st.session_state.user_type = login_result["user_type"]
                
                # Simulate loading
                with st.spinner("Redirecting..."):
                    time.sleep(1)
                
                # If user is a regular user, redirect to salary prediction
                if st.session_state.role == "User":
                    st.info("Redirecting to salary prediction application...")
                    st.session_state.page = 'salary_prediction'
                else:
                    st.session_state.page = 'dashboard'
                st.rerun()
            else:
                st.error(login_result["message"])

def display_dashboard():
    st.header(f"Welcome, {st.session_state.username}!")
    
    if st.session_state.role == "Admin":
        st.subheader("Admin Dashboard")
        st.write("This is the admin dashboard. You have access to all features.")
        
        # Admin-specific functionality could go here
        st.write("Admin features:")
        st.write("- User management")
        st.write("- System configuration")
        st.write("- Analytics and reporting")
        
        # Button for admins to access the salary prediction app
        if st.button("Access Salary Prediction Tool"):
            st.session_state.page = 'salary_prediction'
            st.rerun()
    else:
        st.subheader(f"{st.session_state.user_type} Dashboard")
        st.write(f"This is the {st.session_state.user_type.lower()} dashboard.")
        
        # Button for users to access the salary prediction app
        if st.button("Access Salary Prediction Tool"):
            st.session_state.page = 'salary_prediction'
            st.rerun()
    
    if st.button("Logout"):
        # Reset session state
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.user_type = None
        st.rerun()

def display_salary_prediction():
    # Display user info in sidebar
    st.sidebar.success(f"Logged in as: {st.session_state.username}")
    st.sidebar.info(f"Role: {st.session_state.role}")
    if hasattr(st.session_state, 'user_type') and st.session_state.user_type:
        st.sidebar.info(f"Account type: {st.session_state.user_type}")
    
    # Navigation buttons in sidebar
    if st.sidebar.button("Back to Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    if st.sidebar.button("Logout"):
        # Reset session state
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.user_type = None
        st.rerun()

    # Load model, scaler, and encoders
    def load_artifacts():
        try:
            model = joblib.load('salary_model.pkl')
            scaler = joblib.load('scaler.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            return model, scaler, label_encoders
        except FileNotFoundError as e:
            st.error(f"Model files not found: {e}")
            st.info("This is a demo application. Please ensure model files are available.")
            return None, None, None

    # Feature options
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    job_roles = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
    locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'India']
    genders = ['Male', 'Female', 'Other']

    st.title('Employee Salary Prediction')
    st.write('Predict employee salary using manual input or batch CSV upload.')
    st.write('Developed by T Jagadish.')

    try:
        model, scaler, label_encoders = load_artifacts()
        
        if model is not None:
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
        else:
            st.warning("Model files are missing. This is a demonstration of the interface.")
            st.info("In a real deployment, ensure the model files are properly loaded.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("This is likely because the model files (salary_model.pkl, scaler.pkl, label_encoders.pkl) are not available.")
        st.info("For demonstration purposes, you can still see the interface structure.")

def validate_login(username, password, role, user_type=None):
    """
    Validates user credentials against the predefined USERS dictionary
    """
    if username in USERS:
        user_data = USERS[username]
        
        # Check if the role matches
        if user_data["role"] != role:
            return {
                "success": False, 
                "message": f"This account is a {user_data['role']} account, not a {role} account."
            }
        
        # Check password
        if user_data["password"] == password:
            # For User role, check user_type if applicable
            if role == "User":
                actual_user_type = user_data.get("user_type", user_type)
                return {"success": True, "user_type": actual_user_type}
            return {"success": True, "user_type": None}
        else:
            return {"success": False, "message": "Invalid password"}
    else:
        # For non-existing users, allow login with any password for demo purposes
        # In a real application, you would reject unknown users
        if role == "User" and username and password:
            return {"success": True, "user_type": user_type}
        return {"success": False, "message": "Invalid username or password"}
    
if __name__ == "__main__":
    # Set page configuration
    st.set_page_config(
        page_title="Employee System",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add some custom CSS for better styling
    st.markdown("""
        <style>
        .stButton button {
            height: 3rem;
            font-size: 1.2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    main()