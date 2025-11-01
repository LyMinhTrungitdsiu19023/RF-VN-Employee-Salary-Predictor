import streamlit as st
import time
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

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
    elif st.session_state.page == 'admin_user_management':
        display_admin_user_management()
    elif st.session_state.page == 'admin_data_visualization':
        display_admin_data_visualization()
    elif st.session_state.page == 'admin_model_management':
        display_admin_model_management()

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
        st.write("Welcome to the admin dashboard. Select a management area below:")
        
        # Create three columns for admin options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("User Management")
            st.write("Manage user accounts and permissions")
            if st.button("User Management", key="user_mgmt_btn", use_container_width=True):
                st.session_state.page = 'admin_user_management'
                st.rerun()
        
        with col2:
            st.info("Data Visualization")
            st.write("View charts and analytics on employee data")
            if st.button("Data Visualization", key="data_viz_btn", use_container_width=True):
                st.session_state.page = 'admin_data_visualization'
                st.rerun()
        
        with col3:
            st.info("Model Management")
            st.write("Adjust model parameters and retrain")
            if st.button("Model Management", key="model_mgmt_btn", use_container_width=True):
                st.session_state.page = 'admin_model_management'
                st.rerun()
        
        st.markdown("---")
        
        # Quick access to salary prediction tool
        if st.button("Access Salary Prediction Tool", use_container_width=True):
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

def display_admin_user_management():
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("User Management")
    st.write("Manage user accounts and permissions")
    
    # Display current users
    st.subheader("Current Users")
    
    # Convert user dictionary to DataFrame for better display
    user_data = []
    for username, details in USERS.items():
        user_dict = {
            "Username": username,
            "Role": details["role"],
            "User Type": details.get("user_type", "N/A")
        }
        user_data.append(user_dict)
    
    user_df = pd.DataFrame(user_data)
    st.dataframe(user_df, use_container_width=True)
    
    # Add new user section
    st.subheader("Add New User")
    
    with st.form("add_user_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["Admin", "User"])
        
        # Only show user type if role is User
        new_user_type = None
        if new_role == "User":
            new_user_type = st.selectbox("User Type", ["Employer", "Job Seeker"])
        
        submit_button = st.form_submit_button("Add User")
        
        if submit_button:
            if new_username and new_password:
                if new_username in USERS:
                    st.error(f"Username '{new_username}' already exists!")
                else:
                    # In a real app, you would add to a database
                    # For this demo, we'll just show a success message
                    st.success(f"User '{new_username}' would be added in a real application.")
                    
                    # Display what would be added
                    new_user_data = {
                        "password": new_password,
                        "role": new_role
                    }
                    if new_role == "User" and new_user_type:
                        new_user_data["user_type"] = new_user_type
                    
                    st.code(f"{new_username}: {new_user_data}")
            else:
                st.error("Username and password are required!")
    
    # Edit/Delete user section
    st.subheader("Edit/Delete User")
    selected_user = st.selectbox("Select User", list(USERS.keys()))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Edit User", use_container_width=True):
            st.info(f"In a real application, you would be able to edit user '{selected_user}'")
    
    with col2:
        if st.button("Delete User", use_container_width=True):
            st.warning(f"In a real application, user '{selected_user}' would be deleted")

def display_admin_data_visualization():
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Data Visualization")
    st.write("Explore employee salary data through visualizations")
    
    # Try to load the employee salary data
    try:
        # First try to load from employee_salary_data.csv
        if os.path.exists('employee_salary_data.csv'):
            df = pd.read_csv('employee_salary_data.csv')
            st.success("Successfully loaded employee salary data")
        else:
            # If file doesn't exist, create sample data
            st.warning("employee_salary_data.csv not found. Using sample data for demonstration.")
            df = create_sample_data()
        
        # Display the data
        st.subheader("Employee Salary Data")
        st.dataframe(df.head(10), use_container_width=True)
        st.info(f"Total records: {len(df)}")
        
        # Data summary
        st.subheader("Data Summary")
        st.write(df.describe())
        
        # Visualization options
        st.subheader("Visualizations")
        
        viz_type = st.selectbox(
            "Select Visualization", 
            ["Salary Distribution", "Salary by Experience", "Salary by Education", 
             "Salary by Job Role", "Salary by Location", "Correlation Heatmap"]
        )
        
        if viz_type == "Salary Distribution":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['Salary'], kde=True, ax=ax)
            ax.set_title('Salary Distribution')
            ax.set_xlabel('Salary (₹)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            
        elif viz_type == "Salary by Experience":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='YearsExperience', y='Salary', data=df, ax=ax)
            ax.set_title('Salary vs. Years of Experience')
            ax.set_xlabel('Years of Experience')
            ax.set_ylabel('Salary (₹)')
            st.pyplot(fig)
            
        elif viz_type == "Salary by Education":
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Education', y='Salary', data=df, ax=ax)
            ax.set_title('Salary by Education Level')
            ax.set_xlabel('Education Level')
            ax.set_ylabel('Salary (₹)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        elif viz_type == "Salary by Job Role":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='JobRole', y='Salary', data=df, ax=ax)
            ax.set_title('Average Salary by Job Role')
            ax.set_xlabel('Job Role')
            ax.set_ylabel('Average Salary (₹)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        elif viz_type == "Salary by Location":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Location', y='Salary', data=df, ax=ax)
            ax.set_title('Average Salary by Location')
            ax.set_xlabel('Location')
            ax.set_ylabel('Average Salary (₹)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        elif viz_type == "Correlation Heatmap":
            # Create a copy of the dataframe with numeric columns only
            numeric_df = df.select_dtypes(include=[np.number])
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
        
        # Custom visualization
        st.subheader("Custom Visualization")
        st.write("Create your own visualization by selecting variables to plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("X-axis", df.columns)
        
        with col2:
            y_var = st.selectbox("Y-axis", df.columns, index=1)
        
        plot_type = st.selectbox("Plot Type", ["Scatter", "Bar", "Line", "Box"])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "Scatter":
            sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax)
        elif plot_type == "Bar":
            sns.barplot(x=x_var, y=y_var, data=df, ax=ax)
        elif plot_type == "Line":
            sns.lineplot(x=x_var, y=y_var, data=df, ax=ax)
        elif plot_type == "Box":
            sns.boxplot(x=x_var, y=y_var, data=df, ax=ax)
        
        ax.set_title(f'{y_var} by {x_var}')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading or visualizing data: {e}")
        st.info("Please ensure the employee_salary_data.csv file exists and is properly formatted.")

def display_admin_model_management():
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Model Management")
    st.write("Adjust model parameters and retrain the salary prediction model")
    
    # First try to load from employee_salary_data.csv
    if os.path.exists('employee_salary_data.csv'):
        df = pd.read_csv('employee_salary_data.csv')
        st.success("Successfully loaded employee salary data for model training")
    else:
        # If file doesn't exist, create sample data
        st.warning("employee_salary_data.csv not found. Using sample data for demonstration.")
        df = create_sample_data()
        
    # Display data sample
    st.subheader("Training Data Sample")
    st.dataframe(df.head(), use_container_width=True)
    
    # Model parameters section
    st.subheader("Random Forest Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 10)
        max_depth = st.slider("Max Depth", 5, 50, 10, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, 1)
    
    with col2:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1, 1)
        max_features = st.selectbox("Max Features", ["auto", "sqrt", "log2"])
        random_state = st.number_input("Random State", 0, 100, 42, 1)
    
    # Train-test split parameters
    st.subheader("Train-Test Split")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # Button to train model
    if st.button("Train Model", use_container_width=True):
        with st.spinner("Training model..."):
            # Prepare data
            X = df.drop('Salary', axis=1)
            y = df['Salary']
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Display metrics
            st.subheader("Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Squared Error", f"{mse:.2f}")
            
            with col2:
                st.metric("Root Mean Squared Error", f"{rmse:.2f}")
            
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
            
            # Save model and artifacts
            st.subheader("Save Model")
            
            if st.button("Save Model and Artifacts"):
                try:
                    joblib.dump(model, 'salary_model.pkl')
                    joblib.dump(scaler, 'scaler.pkl')
                    joblib.dump(label_encoders, 'label_encoders.pkl')
                    st.success("Model and artifacts saved successfully!")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
    
    # Data upload section for adding new training data
    st.subheader("Add Training Data")
    st.write("Upload a CSV file with additional training data")
    
    upload_file = st.file_uploader("Upload CSV", type=['csv'])
    if upload_file is not None:
        try:
            new_data = pd.read_csv(upload_file)
            st.write("Preview of uploaded data:")
            st.dataframe(new_data.head(), use_container_width=True)
            
            if st.button("Add to Training Data"):
                # In a real app, you would validate and merge with existing data
                st.success("Data would be added to training dataset in a real application.")
                st.info(f"Uploaded {len(new_data)} records.")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")


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

    # Feature options
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    job_roles = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
    locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'India']
    genders = ['Male', 'Female', 'Other']

    st.title('Employee Salary Prediction')
    st.write('Predict employee salary using manual input or batch CSV upload.')
    st.write('Developed by T Jagadish.')

    try:
        # Load model, scaler, and encoders
        model, scaler, label_encoders = None, None, None
        try:
            model = joblib.load('salary_model.pkl')
            scaler = joblib.load('scaler.pkl')
            label_encoders = joblib.load('label_encoders.pkl')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load model: {e}")
            st.info("Using demo mode with mock predictions.")
        
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
                if model is not None and scaler is not None and label_encoders is not None:
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
                else:
                    # Mock prediction for demo
                    base_salary = 50000
                    exp_factor = experience * 5000
                    
                    edu_factor = 0
                    if education == 'Bachelor':
                        edu_factor = 20000
                    elif education == 'Master':
                        edu_factor = 40000
                    elif education == 'PhD':
                        edu_factor = 60000
                    
                    role_factor = 0
                    if job_role == 'Data Scientist':
                        role_factor = 30000
                    elif job_role == 'Software Engineer':
                        role_factor = 25000
                    elif job_role == 'Manager':
                        role_factor = 35000
                    elif job_role == 'Analyst':
                        role_factor = 15000
                    elif job_role == 'HR':
                        role_factor = 10000
                    
                    loc_factor = 0
                    if location == 'New York':
                        loc_factor = 30000
                    elif location == 'San Francisco':
                        loc_factor = 35000
                    elif location == 'Austin':
                        loc_factor = 15000
                    elif location == 'Remote':
                        loc_factor = 10000
                    elif location == 'India':
                        loc_factor = 5000
                    
                    age_factor = (age - 22) * 1000
                    
                    # Add some random noise
                    import random
                    noise = random.uniform(-10000, 10000)
                    
                    # Calculate mock salary
                    mock_salary = base_salary + exp_factor + edu_factor + role_factor + loc_factor + age_factor + noise
                    st.success(f'Predicted Salary (Demo): ₹{int(mock_salary):,.0f}')
                    st.info("Note: This is a mock prediction as model files are not available.")

        # --- Batch Prediction ---
        st.header('Batch Prediction (CSV Upload)')
        st.write('Upload a CSV file with columns: YearsExperience, Education, JobRole, Location, Age, Gender')
        file = st.file_uploader('Upload CSV', type=['csv'])
        if file is not None:
            try:
                df_upload = pd.read_csv(file)
                # Check columns
                expected_cols = ['YearsExperience', 'Education', 'JobRole', 'Location', 'Age', 'Gender']
                if all(col in df_upload.columns for col in expected_cols):
                    if model is not None and scaler is not None and label_encoders is not None:
                        # Encode categorical columns
                        for col in ['Education', 'JobRole', 'Location', 'Gender']:
                            df_upload[col] = label_encoders[col].transform(df_upload[col])
                        X_upload = df_upload[expected_cols]
                        X_upload_scaled = scaler.transform(X_upload)
                        salary_preds = model.predict(X_upload_scaled)
                        df_upload['PredictedSalary'] = salary_preds.astype(int)
                    else:
                        # Mock batch predictions
                        st.info("Using mock predictions for demonstration.")
                        mock_salaries = []
                        for _, row in df_upload.iterrows():
                            base = 50000
                            exp_factor = row['YearsExperience'] * 5000
                            
                            edu_factor = 0
                            if row['Education'] == 'Bachelor':
                                edu_factor = 20000
                            elif row['Education'] == 'Master':
                                edu_factor = 40000
                            elif row['Education'] == 'PhD':
                                edu_factor = 60000
                            
                            role_factor = 0
                            if row['JobRole'] == 'Data Scientist':
                                role_factor = 30000
                            elif row['JobRole'] == 'Software Engineer':
                                role_factor = 25000
                            elif row['JobRole'] == 'Manager':
                                role_factor = 35000
                            elif row['JobRole'] == 'Analyst':
                                role_factor = 15000
                            elif row['JobRole'] == 'HR':
                                role_factor = 10000
                            
                            loc_factor = 0
                            if row['Location'] == 'New York':
                                loc_factor = 30000
                            elif row['Location'] == 'San Francisco':
                                loc_factor = 35000
                            elif row['Location'] == 'Austin':
                                loc_factor = 15000
                            elif row['Location'] == 'Remote':
                                loc_factor = 10000
                            elif row['Location'] == 'India':
                                loc_factor = 5000
                            
                            age_factor = (row['Age'] - 22) * 1000
                            
                            # Add some random noise
                            import random
                            noise = random.uniform(-10000, 10000)
                            
                            mock_salary = base + exp_factor + edu_factor + role_factor + loc_factor + age_factor + noise
                            mock_salaries.append(int(mock_salary))
                        
                        df_upload['PredictedSalary'] = mock_salaries
                    
                    st.dataframe(df_upload)
                    # Download link
                    csv = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button('Download Predictions as CSV', csv, 'salary_predictions.csv', 'text/csv')
                else:
                    st.error(f'CSV must contain columns: {expected_cols}')
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")
                st.info("Please ensure your CSV file is properly formatted.")

        # --- (Optional) Model Metrics ---
        st.header('Model & Dataset Info')
        st.write('Trained on 500 synthetic employee records. Model: Random Forest Regressor.')
        st.write('Features: YearsExperience, Education, JobRole, Location, Age, Gender')
        st.write('All salaries are shown in Indian Rupees (₹).')
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("The application is running in a limited mode. Some features may not be available.")

def display_admin_sidebar():
    """Display the admin navigation sidebar"""
    st.sidebar.success(f"Logged in as: {st.session_state.username} (Admin)")
    
    st.sidebar.title("Admin Navigation")
    
    if st.sidebar.button("Dashboard"):
        st.session_state.page = 'dashboard'
        st.rerun()
        
    if st.sidebar.button("User Management"):
        st.session_state.page = 'admin_user_management'
        st.rerun()
        
    if st.sidebar.button("Data Visualization"):
        st.session_state.page = 'admin_data_visualization'
        st.rerun()
        
    if st.sidebar.button("Model Management"):
        st.session_state.page = 'admin_model_management'
        st.rerun()
        
    if st.sidebar.button("Salary Prediction Tool"):
        st.session_state.page = 'salary_prediction'
        st.rerun()
        
    if st.sidebar.button("Logout"):
        # Reset session state
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.user_type = None
        st.rerun()

def create_sample_data(n_samples=500):
    """Create sample employee salary data for demonstration"""
    np.random.seed(42)
    
    # Generate features
    years_exp = np.random.uniform(0, 20, n_samples)
    age = np.random.randint(22, 60, n_samples)
    
    # Categorical features
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    job_roles = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
    locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'India']
    genders = ['Male', 'Female', 'Other']
    
    education = np.random.choice(education_levels, n_samples)
    job_role = np.random.choice(job_roles, n_samples)
    location = np.random.choice(locations, n_samples)
    gender = np.random.choice(genders, n_samples)
    
    # Generate salary based on features
    base_salary = 50000
    exp_factor = years_exp * 5000
    
    edu_factor = np.zeros(n_samples)
    edu_factor[education == 'Bachelor'] = 20000
    edu_factor[education == 'Master'] = 40000
    edu_factor[education == 'PhD'] = 60000
    
    role_factor = np.zeros(n_samples)
    role_factor[job_role == 'Data Scientist'] = 30000
    role_factor[job_role == 'Software Engineer'] = 25000
    role_factor[job_role == 'Manager'] = 35000
    role_factor[job_role == 'Analyst'] = 15000
    role_factor[job_role == 'HR'] = 10000
    
    loc_factor = np.zeros(n_samples)
    loc_factor[location == 'New York'] = 30000
    loc_factor[location == 'San Francisco'] = 35000
    loc_factor[location == 'Austin'] = 15000
    loc_factor[location == 'Remote'] = 10000
    loc_factor[location == 'India'] = 5000
    
    age_factor = (age - 22) * 1000
    
    # Add some random noise
    noise = np.random.normal(0, 10000, n_samples)
    
    # Calculate final salary
    salary = base_salary + exp_factor + edu_factor + role_factor + loc_factor + age_factor + noise
    salary = np.round(salary).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'YearsExperience': years_exp,
        'Education': education,
        'JobRole': job_role,
        'Location': location,
        'Age': age,
        'Gender': gender,
        'Salary': salary
    })
    
    return df

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
        .admin-card {
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    main()