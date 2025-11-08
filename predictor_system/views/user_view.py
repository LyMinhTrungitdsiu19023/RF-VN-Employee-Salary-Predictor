import streamlit as st
import time
from controllers.auth_controller import AuthController
from controllers.user_controller import UserController
from config.settings import EDUCATION_LEVELS, JOB_ROLES, GENDERS, RATE_INR_TO_VND, CURRENCY_SYMBOL_VND

def display_salary_prediction():
    """Display the salary prediction page"""
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
        AuthController.logout()
        st.rerun()

    st.title('Employee Salary Prediction')
    st.write('Predict employee salary using manual input or batch CSV upload.')

    try:
        # Get active model version
        active_version = st.session_state.get('active_model_version')
        if not active_version:
            st.warning("No active model set. Using demo mode.")
        else:
            st.info(f"Using active model (Version {active_version})")

        # --- Manual Input ---
        st.header('Manual Prediction')
        with st.form('manual_form'):
            experience = st.slider('Years of Experience', 0, 20, 5)
            education = st.selectbox('Education Level', EDUCATION_LEVELS)
            job_role = st.selectbox('Job Role', JOB_ROLES)
            age = st.slider('Age', 22, 60, 30)
            gender = st.selectbox('Gender', GENDERS)
            submitted = st.form_submit_button('Predict Salary')

            if submitted:
                # Create features dictionary
                features = {
                    'experience': experience,
                    'education': education,
                    'job_role': job_role,
                    'age': age,
                    'gender': gender
                }
                
                # Get prediction from controller with version
                salary_pred = UserController.predict_salary(features, model_version=active_version)
                
                # Convert prediction to VND for display
                try:
                    vnd_salary = float(salary_pred) * RATE_INR_TO_VND
                except Exception:
                    vnd_salary = salary_pred

                # Display prediction with nice formatting (VND)
                st.markdown(f"""
                    <div style='background-color: #e6f3ff; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #0066cc; text-align: center;'>Predicted Salary</h3>
                        <h2 style='color: #004d99; text-align: center;'>{CURRENCY_SYMBOL_VND} {vnd_salary:,.0f}</h2>
                    </div>
                """, unsafe_allow_html=True)

        # --- Batch Prediction (Only for Recruiters) ---
        if hasattr(st.session_state, 'user_type') and st.session_state.user_type == "Recruiter":
            st.header('Batch Prediction (CSV Upload)')
            st.write('Upload a CSV file with columns: YearsExperience, Education, JobRole, Age, Gender')
            file = st.file_uploader('Upload CSV', type=['csv'])
            if file is not None:
                # Process batch prediction
                result = UserController.predict_batch(file, model_version=active_version)
                if result["success"]:
                    # Convert PredictedSalary to VND for display and download if present
                    df_out = result["data"].copy()
                    if 'PredictedSalary' in df_out.columns:
                        try:
                            df_out['PredictedSalary'] = (df_out['PredictedSalary'].astype(float) * RATE_INR_TO_VND).round().astype(int)
                        except Exception:
                            pass

                    st.dataframe(df_out)
                    # Download link
                    csv = df_out.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Predictions as CSV', csv, 'salary_predictions_vnd.csv', 'text/csv')
                else:
                    st.error(result["message"])
                    st.info("Please ensure your CSV file is properly formatted.")
        elif hasattr(st.session_state, 'user_type') and st.session_state.user_type != "Recruiter":
            st.info("Batch prediction is only available for Recruiter accounts.")
            
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("The application is running in a limited mode. Some features may not be available.")