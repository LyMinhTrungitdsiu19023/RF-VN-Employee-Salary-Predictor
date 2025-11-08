import streamlit as st
import time
import pandas as pd
from controllers.auth_controller import AuthController
from controllers.admin_controller import AdminController
from config.settings import RATE_INR_TO_VND, CURRENCY_SYMBOL_VND

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
        AuthController.logout()
        st.rerun()

def display_dashboard():
    """Display the admin or user dashboard"""
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
        # Check if user_type exists and is not None
        user_type = st.session_state.get('user_type', 'User')
        st.subheader(f"{user_type} Dashboard")
        st.write(f"This is the {user_type.lower() if user_type else 'user'} dashboard.")
        
        # Button for users to access the salary prediction app
        if st.button("Access Salary Prediction Tool"):
            st.session_state.page = 'salary_prediction'
            st.rerun()
    
    if st.button("Logout"):
        AuthController.logout()
        st.rerun()

def display_admin_user_management():
    """Display the admin user management page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("User Management")
    st.write("Manage user accounts and permissions")
    
    # Display current users
    st.subheader("Current Users")
    
    # Get user data from controller
    user_data = AdminController.get_all_users()
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
            new_user_type = st.selectbox("User Type", ["Student", "Software Engineer", "Recruiter"])
        
        submit_button = st.form_submit_button("Add User")
        
        if submit_button:
            result = AdminController.add_user(new_username, new_password, new_role, new_user_type)
            if result["success"]:
                st.success(result["message"])
                st.code(f"{new_username}: {result['user_data']}")
            else:
                st.error(result["message"])

def display_admin_data_visualization():
    """Display the admin data visualization page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Data Visualization")
    st.write("Explore employee salary data through visualizations")
    
    try:
        # Get data from controller
        df = AdminController.get_employee_data()
        
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
        
        # Create and display visualization
        fig = AdminController.create_visualization(df, viz_type)
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
        
        # Create and display custom visualization
        fig = AdminController.create_visualization(df, "Custom Plot", x_var, y_var, plot_type)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error loading or visualizing data: {e}")
        st.info("Please ensure the employee_salary_data.csv file exists and is properly formatted.")

def display_admin_model_management():
    """Display the admin model management page"""
    # Admin navigation sidebar
    display_admin_sidebar()
    
    st.title("Model Management")
    
    # Get available model versions
    versions = AdminController.get_available_versions()
    
    # Model version selection
    st.subheader("Model Version Selection")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if versions:
            st.info(f"Available model versions: {len(versions)}")
            version_list = ['New Version'] + [f'Version {v}' for v in versions]
            selected_version_str = st.selectbox(
                "Select Model Version",
                options=version_list,
                index=0
            )
            selected_version = None if selected_version_str == 'New Version' else int(selected_version_str.split()[-1])
        else:
            st.warning("No trained models available. Creating new version.")
            selected_version = None
            
    with col2:
        if selected_version:
            if st.button("Set as Active", use_container_width=True):
                st.session_state.active_model_version = selected_version
                st.success(f"Set version {selected_version} as active model")
                st.rerun()
                
    # Display active model version
    if hasattr(st.session_state, 'active_model_version'):
        st.success(f"Active model version: {st.session_state.active_model_version}")
    
    # Get data from controller
    df = AdminController.get_employee_data()
        
    # Display data sample
    st.subheader("Training Data Sample")
    # Also show a VND-converted salary column for clarity (original dataframe remains unchanged)
    df_display = df.copy()
    if 'Salary' in df_display.columns:
        try:
            df_display['Salary_VND'] = (df_display['Salary'].astype(float) * RATE_INR_TO_VND).round().astype(int)
        except Exception:
            pass
    st.dataframe(df_display.head(), use_container_width=True)
    
    # Model parameters section
    st.subheader("Random Forest Model Parameters")
    
    # If a version is selected, get its parameters
    if selected_version:
        model_info = AdminController.get_model_parameters(selected_version)
        if model_info:
            st.info(f"Showing parameters for Version {selected_version}")
            if 'created_at' in model_info:
                st.write(f"Created: {model_info['created_at']}")
            if 'feature_names' in model_info:
                st.write(f"Features: {', '.join(model_info['feature_names'])}")
            params = model_info.get('parameters', model_info)
        else:
            st.error(f"Could not load parameters for Version {selected_version}")
            params = None
    else:
        params = None
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 
            params.get('n_estimators', 100) if params else 100, 10)
        max_depth = st.slider("Max Depth", 5, 50, 
            params.get('max_depth', 10) if params else 10, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 
            params.get('min_samples_split', 2) if params else 2, 1)
    
    with col2:
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 
            params.get('min_samples_leaf', 1) if params else 1, 1)
        max_features = st.selectbox("Max Features", ["sqrt", "log2", None],
            index=["sqrt", "log2", None].index(params.get('max_features', 'sqrt')) if params else 0)
        random_state = st.number_input("Random State", 0, 100, 
            params.get('random_state', 42) if params else 42, 1)
    
    # Train-test split parameters
    st.subheader("Train-Test Split")
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    
    # Initialize metrics variable
    metrics = None
    
    # Model management buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        train_button_text = "Train New Model" if not selected_version else f"Retrain Version {selected_version}"
        if st.button(train_button_text, use_container_width=True):
            with st.spinner("Training model..."):
                # Prepare model parameters
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'random_state': random_state,
                    'test_size': test_size
                }
                
                # Train model and get metrics
                metrics = AdminController.train_model(df, params, save_as_version=selected_version)
                
                # Check if training was successful
                if metrics and metrics.get('success', False):
                    # Show success message
                    st.success(metrics['message'])
                    
                    # Show model version info
                    if 'version' in metrics:
                        st.info(f"Model version: v{metrics['version']}")
                        
                        # Clear any cached version data
                        if "version_list" in st.session_state:
                            del st.session_state.version_list
                            
                        # If no active version is set, set this as active
                        if not st.session_state.get('active_model_version'):
                            st.session_state.active_model_version = metrics['version']
                            st.info(f"Set as active model version")
                    
                    # Display metrics
                    st.subheader("Model Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Squared Error", f"{metrics['mse']:.2f}")
                    
                    with col2:
                        st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
                    
                    with col3:
                        st.metric("R² Score", f"{metrics['r2']:.4f}")
                    
                    # Feature importance
                    st.subheader("Feature Importance")
                    st.dataframe(metrics['feature_importance'], use_container_width=True)
                    
                    # Plot feature importance
                    if 'feature_importance' in metrics:
                        fig = AdminController.create_visualization(
                            metrics['feature_importance'], 
                            "Custom Plot", 
                            "Importance", 
                            "Feature", 
                            "Bar"
                        )
                        st.pyplot(fig)
                    
                    # Force refresh after short delay
                    time.sleep(1)
                    st.rerun()
                elif metrics:
                    st.error("Failed to save model. Check if you have write permissions to the model directory.")
    
    with col2:
        if selected_version and selected_version_str != 'New Version':
            # Store delete confirmation state in session state
            if 'delete_confirm' not in st.session_state:
                st.session_state.delete_confirm = False
                
            if not st.session_state.delete_confirm:
                if st.button("Delete Version", use_container_width=True, type="secondary", key="delete_btn"):
                    if st.session_state.get('active_model_version') == selected_version:
                        st.error("Cannot delete the active model version. Please set another version as active first.")
                    else:
                        st.session_state.delete_confirm = True
                        st.rerun()
            else:
                st.warning(f"Are you sure you want to delete Version {selected_version}?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Delete", type="primary", use_container_width=True):
                        success = AdminController.delete_model_version(selected_version)
                        if success:
                            st.success(f"Version {selected_version} deleted successfully")
                            # Clear session states
                            st.session_state.delete_confirm = False
                            if "version_list" in st.session_state:
                                del st.session_state.version_list
                            # Force refresh versions
                            versions = AdminController.get_available_versions()
                            if not versions:
                                st.session_state.page = 'admin_model_management'
                            st.rerun()
                        else:
                            st.error("Failed to delete model version")
                with col2:
                    if st.button("Cancel", type="secondary", use_container_width=True):
                        st.session_state.delete_confirm = False
                        st.rerun()
            
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