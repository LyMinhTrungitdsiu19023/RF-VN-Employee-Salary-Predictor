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
            df_display['Salary_VND'] = (df_display['Salary'].astype(float)).round().astype(int)
        except Exception:
            pass
    st.dataframe(df_display.head(50), use_container_width=True)
    
    # Initialize params with default values for new version
    params = None
    
    # If a version is selected, get its parameters
    if selected_version:
        model_info = AdminController.get_model_parameters(selected_version)
        if model_info:
            st.info(f"Showing parameters for Version {selected_version}")
            if 'created_at' in model_info:
                st.write(f"Created: {model_info['created_at']}")
            if 'feature_names' in model_info:
                st.write(f"Features: {', '.join(model_info['feature_names'])}")
            
            # Display if model uses engineered features
            if model_info.get('uses_engineered_features'):
                st.success("This model uses engineered features for better predictions")
            
            # Display if model uses log transformation
            if model_info.get('is_log_transformed'):
                st.info("This model uses log transformation for the target variable")
            
            # Display metrics if available
            if 'metrics' in model_info and model_info['metrics']:
                st.subheader("Model Performance Metrics")
                metrics = model_info['metrics']

                def format_metric(value, decimal_places=2):
                    """Format a metric value, handling both numbers and strings"""
                    if isinstance(value, (int, float)):
                        return f"{value:.{decimal_places}f}"
                    return "N/A"
                
                # Determine if this model was tuned with cross-validation
                is_tuned_model = 'best_score' in metrics or 'cv_folds' in metrics or 'tuning_method' in metrics
                
                # First row of metrics - prioritize CV metrics for tuned models
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # For tuned models, show CV RMSE first if available
                    if is_tuned_model and metrics.get('best_score') is not None and metrics.get('scoring', '').startswith('neg_'):
                        cv_rmse = abs(metrics.get('best_score', 0))  # Convert from negative to positive
                        st.metric("CV RMSE", format_metric(cv_rmse, 4), 
                                help="Cross-validation RMSE - more reliable for tuned models")
                        import math
                        # Calculate percentage error based on CV RMSE if it's log-transformed
                        if model_info.get('is_log_transformed'):
                            percent_error = (math.exp(cv_rmse) - 1) * 100
                            st.metric("% Model Error", f"{format_metric(percent_error, 1)}%", 
                                    help="Approximate percentage error: e^(RMSE Log) - 1")
                    else:
                        # Fall back to regular RMSE
                        st.metric("RMSE", format_metric(metrics.get('rmse', 'N/A')),
                                help="Root Mean Squared Error on test set")
                
                with col2:
                    st.metric("R² Score", format_metric(metrics.get('r2', 'N/A'), 4),
                            help="Coefficient of determination (1.0 is perfect)")
                
                with col3:
                    st.metric("MAE", format_metric(metrics.get('mae', 'N/A')),
                            help="Mean Absolute Error")
                
                # Second row of metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mape_value = metrics.get('mape', 'N/A')
                    mape_display = f"{format_metric(mape_value)}%" if isinstance(mape_value, (int, float)) else "N/A"
                    st.metric("MAPE", mape_display,
                            help="Mean Absolute Percentage Error")
                
                with col2:
                    # For tuned models that don't use CV RMSE as primary metric, show it here
                    if is_tuned_model and not metrics.get('scoring', '').startswith('neg_'):
                        # If we have best_score but it's not RMSE, show the CV score with proper label
                        cv_score = metrics.get('best_score', 'N/A')
                        if isinstance(cv_score, (int, float)):
                            if metrics.get('scoring', '').startswith('neg_'):
                                cv_score = abs(cv_score)  # Convert negative scores to positive
                            scoring_name = metrics.get('scoring', '').replace('neg_', '').replace('_', ' ').title()
                            st.metric(f"CV {scoring_name}", format_metric(cv_score, 4),
                                    help="Cross-validation score - more reliable for tuned models")
                        else:
                            # If no CV score, show regular RMSE
                            st.metric("RMSE", format_metric(metrics.get('rmse', 'N/A')),
                                    help="Root Mean Squared Error on test set")
                    else:
                        # Show log-space RMSE if available
                        if metrics.get('log_rmse') is not None:
                            st.metric("Log-space RMSE", format_metric(metrics.get('log_rmse', 'N/A'), 4),
                                    help="RMSE calculated in logarithmic space")
                        else:
                            # Otherwise show MSE
                            st.metric("MSE", format_metric(metrics.get('mse', 'N/A')),
                                    help="Mean Squared Error")
                
                with col3:
                    # Display median-based metrics if available
                    if metrics.get('med_ape') is not None:
                        med_ape_value = metrics.get('med_ape', 'N/A')
                        med_ape_display = f"{format_metric(med_ape_value)}%" if isinstance(med_ape_value, (int, float)) else "N/A"
                        st.metric("Median APE", med_ape_display,
                                help="Median Absolute Percentage Error - less sensitive to outliers than MAPE")
                    elif is_tuned_model:
                        # For tuned models, show tuning method
                        tuning_method = metrics.get('tuning_method', 'Cross-Validation')
                        cv_folds = metrics.get('cv_folds', 'N/A')
                        st.metric("CV Folds", format_metric(cv_folds, 0) if isinstance(cv_folds, (int, float)) else cv_folds,
                                help=f"Number of cross-validation folds used in {tuning_method}")
                    else:
                        # For non-tuned models, show test set size if available
                        test_size = metrics.get('test_size', 'N/A')
                        if isinstance(test_size, (int, float)):
                            test_size_pct = test_size * 100 if test_size < 1 else test_size
                            st.metric("Test Set %", f"{format_metric(test_size_pct, 0)}%",
                                    help="Percentage of data used for testing")
                
                # Add a note about which metric to focus on
                if is_tuned_model:
                    st.info("📊 **For hyperparameter-tuned models, the CV RMSE (or other CV metric) is generally more reliable than the test set RMSE.**")
                
                # Add explanation for metrics if needed
                with st.expander("Metrics Explanation"):
                    st.write("""
                    - **CV RMSE**: Cross-validation Root Mean Squared Error - More reliable for tuned models
                    - **% Model Error**: Approximate percentage error of predictions (e^RMSE_log - 1) - For log-transformed models, this represents the typical percentage deviation from true values
                    - **RMSE**: Root Mean Squared Error on test set - Lower is better
                    - **R² Score**: Coefficient of determination - Higher is better (1.0 is perfect)
                    - **MAE**: Mean Absolute Error - Lower is better
                    - **MAPE**: Mean Absolute Percentage Error - Lower is better
                    - **MSE**: Mean Squared Error - Lower is better
                    - **Log-space metrics**: Metrics calculated in logarithmic space (when log transformation is used)
                    - **Median APE**: Median Absolute Percentage Error - Less sensitive to outliers than MAPE
                    """)
                    
                    # Add practical explanation of percentage error
                    if model_info.get('is_log_transformed') and metrics.get('best_score') is not None:
                        cv_rmse = abs(metrics.get('best_score', 0))
                        percent_error = (math.exp(cv_rmse) - 1) * 100
                        st.write(f"""
                        **Practical interpretation of % Model Error ({format_metric(percent_error, 1)}%):**
                        - If the actual salary is 20 million VND, the typical error is about: {format_metric(20 * percent_error / 100, 1)} million VND
                        - If the actual salary is 50 million VND, the typical error is about: {format_metric(50 * percent_error / 100, 1)} million VND
                        """)
                    
                    # Show the scoring metric used if available
                    if 'scoring' in metrics:
                        st.write(f"**Scoring metric used**: {metrics['scoring']}")
                    
                    # Show tuning method if available
                    if 'tuning_method' in metrics:
                        st.write(f"**Tuning method**: {metrics['tuning_method']}")
                        
                    # Show CV folds if available
                    if 'cv_folds' in metrics:
                        st.write(f"**Cross-validation folds**: {metrics['cv_folds']}")
            else:
                st.info("No performance metrics available for this model version.")
    
    # Model parameters section
    st.subheader("Random Forest Model Parameters")
    
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


    st.subheader("Model Version Management")

    if selected_version is not None and selected_version_str != 'New Version':
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Set as Active Model", key="set_active_btn", use_container_width=True):
                st.session_state.active_model_version = selected_version
                st.success(f"Set version {selected_version} as active model")
                st.rerun()
        
        with col2:
            # Store delete confirmation state in session state
            if 'delete_confirm' not in st.session_state:
                st.session_state.delete_confirm = False
                
            if not st.session_state.delete_confirm:
                if st.button("Delete This Version", use_container_width=True, type="secondary", key="delete_version_btn"):
                    if st.session_state.get('active_model_version') == selected_version:
                        st.error("Cannot delete the active model version. Please set another version as active first.")
                    else:
                        st.session_state.delete_confirm = True
                        st.rerun()
            else:
                st.warning(f"Are you sure you want to delete Version {selected_version}?")
                delete_col1, delete_col2 = st.columns(2)
                with delete_col1:
                    if st.button("Yes, Delete", type="primary", use_container_width=True, key="confirm_delete_btn"):
                        success = AdminController.delete_model_version(selected_version)
                        if success:
                            st.success(f"Version {selected_version} deleted successfully")
                            # Clear session states
                            st.session_state.delete_confirm = False
                            if "version_list" in st.session_state:
                                del st.session_state.version_list
                            # Force refresh versions
                            versions = AdminController.get_available_versions()
                            st.session_state.page = 'admin_model_management'
                            st.rerun()
                        else:
                            st.error("Failed to delete model version")
                with delete_col2:
                    if st.button("Cancel", type="secondary", use_container_width=True, key="cancel_delete_btn"):
                        st.session_state.delete_confirm = False
                        st.rerun()
    else:
        st.info("Select an existing model version above to manage it.")

    # Add option for log transformation
    use_log_transform = st.checkbox("Use Log Transformation for Target Variable", value=True)
    
    # Initialize metrics variable
    metrics = None
    
    # Add tabs for different model training options
    model_tabs = st.tabs(["Standard Training (Bad Choice)", "Feature Engineering", "Hyperparameter Tuning", "Data Management"])
    
    with model_tabs[0]:  # Standard Training tab
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
                    if use_log_transform:
                        metrics = AdminController.train_model_with_log_transform(df, params, save_as_version=selected_version)
                    else:
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

    with model_tabs[1]:  # Feature Engineering tab
        st.subheader("Train with Engineered Features")
        st.write("""
        This option creates additional features to improve model performance:
        - Experience transformations (squared, buckets)
        - Age-related features (age-to-experience ratio)
        - Education level as numerical
        - Job role complexity
        - Interaction features (education × experience, role × experience)
        - Experience-to-age ratio
        """)
        
        # Add option for log transformation in this tab too
        use_log_transform_eng = st.checkbox("Use Log Transformation for Target Variable", value=True, key="log_transform_eng")
        
        # Model management buttons
        col1, col2 = st.columns([3, 1])
        with col1:
            train_button_text = "Train New Model with Features" if not selected_version else f"Retrain Version {selected_version} with Features"
            if st.button(train_button_text, use_container_width=True, key="train_engineered"):
                with st.spinner("Training model with engineered features..."):
                    try:
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
                        
                        # Train model with engineered features and get metrics
                        metrics = AdminController.train_model_with_engineered_features(
                            df, 
                            params, 
                            save_as_version=selected_version,
                            use_log_transform=use_log_transform_eng
                        )
                        
                        # Check if training was successful
                        if metrics and metrics.get('success', False):
                            # Show success message
                            st.success(metrics['message'])
                            
                            # Show model version info
                            if 'version' in metrics:
                                st.info(f"Model version: v{metrics['version']} (with engineered features)")
                                
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
                                st.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f}")
                            
                            with col2:
                                st.metric("R² Score", f"{metrics['r2']:.4f}")
                            
                            with col3:
                                st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
                            
                            # Add second row of metrics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("MAPE", f"{metrics['mape']:.2f}%")
                            
                            with col2:
                                if 'oob_score' in metrics and metrics['oob_score'] is not None:
                                    st.metric("Out-of-Bag Score", f"{metrics['oob_score']:.4f}")
                            
                            # Feature importance
                            if 'feature_importance' in metrics:
                                st.subheader("Feature Importance")
                                st.dataframe(metrics['feature_importance'], use_container_width=True)
                                
                                # Plot feature importance
                                fig = AdminController.create_visualization(
                                    metrics['feature_importance'], 
                                    "Custom Plot", 
                                    "Importance", 
                                    "Feature", 
                                    "Bar"
                                )
                                st.pyplot(fig)
                        else:
                            # Display error message
                            st.error(f"Model training failed: {metrics.get('message', 'Unknown error')}")
                            
                    except Exception as e:
                        # Display any exceptions that weren't caught
                        st.error(f"Error during model training: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    with model_tabs[2]:  # Hyperparameter Tuning tab
        st.subheader("Hyperparameter Tuning")
        st.write("Find the optimal hyperparameters for your model using grid search or randomized search.")
        
        # Check if we have tuning results in session state
        if 'tuning_results' in st.session_state and st.session_state.tuning_results:
            # Display saved tuning results
            tuning_results = st.session_state.tuning_results
            
            st.success("Hyperparameter tuning completed successfully!")
            
            # Show best parameters
            st.subheader("Best Parameters")
            st.json(tuning_results['best_params'])
            
            def format_metric(value, decimal_places=2):
                """Format a metric value, handling both numbers and strings"""
                if isinstance(value, (int, float)):
                    return f"{value:.{decimal_places}f}"
                return str(value)
            
            # Show best score
            st.metric("Best Score", format_metric(tuning_results['best_score'], 4))
            
            # Show model version
            if 'version' in tuning_results:
                st.info(f"Model saved as version: v{tuning_results['version']}")
                
                # If no active version is set, set this as active
                if not st.session_state.get('active_model_version'):
                    st.session_state.active_model_version = tuning_results['version']
                    st.info(f"Set as active model version")
            
            # Display CV results visualization if available
            # if 'cv_results_fig' in tuning_results:
            #     st.subheader("Cross-Validation Results")
            #     st.pyplot(tuning_results['cv_results_fig'])
            
            # Add a button to clear results and start over
            if st.button("Clear Results", key="clear_tuning_results"):
                # Clear tuning results from session state
                del st.session_state.tuning_results
                st.rerun()
        else:
            # Select tuning method
            tuning_method = st.radio(
                "Select Tuning Method",
                ["Grid Search", "Randomized Search"],
                horizontal=True
            )
            
            # Configure tuning parameters
            st.subheader("Tuning Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                if tuning_method == "Randomized Search":
                    n_iter = st.slider("Number of Iterations", 10, 100, 20)
            
            with col2:
                scoring = st.selectbox(
                    "Scoring Metric",
                    ["neg_mean_squared_error", "neg_root_mean_squared_error", "r2"]
                )
            
            # Parameter ranges for tuning
            st.subheader("Parameter Ranges")
            
            # n_estimators range
            col1, col2 = st.columns(2)
            with col1:
                n_estimators_min = st.number_input("Min Trees", 10, 500, 50, 10)
            with col2:
                n_estimators_max = st.number_input("Max Trees", n_estimators_min, 1000, 200, 10)
            
            # max_depth range
            col1, col2 = st.columns(2)
            with col1:
                max_depth_min = st.number_input("Min Depth", 3, 50, 5, 1)
            with col2:
                max_depth_max = st.number_input("Max Depth", max_depth_min, 100, 30, 5)
            
            # Include None in max_depth options
            include_none_depth = st.checkbox("Include None in max_depth options (unlimited depth)")
            
            use_engineered_features = st.checkbox("Use Engineered Features", value=True)
            
            # Start tuning button
            if st.button("Start Hyperparameter Tuning", use_container_width=True):
                with st.spinner("Tuning hyperparameters... This may take a while..."):
                    try:
                        # Prepare parameter grid/distribution
                        if tuning_method == "Grid Search":
                            param_grid = {
                                'n_estimators': list(range(n_estimators_min, n_estimators_max + 1, (n_estimators_max - n_estimators_min) // 3)),
                                'max_depth': list(range(max_depth_min, max_depth_max + 1, (max_depth_max - max_depth_min) // 3)),
                                'min_samples_split': [2, 5, 10],
                                'min_samples_leaf': [1, 2, 4],
                                'max_features': ['sqrt', 'log2']
                            }
                            
                            # Add None to max_depth if selected
                            if include_none_depth:
                                param_grid['max_depth'].append(None)
                            
                            # Run grid search with or without engineered features
                            if use_engineered_features:
                                tuning_results = AdminController.tune_model_with_engineered_features(
                                    df, 
                                    method="grid", 
                                    param_grid=param_grid,
                                    cv=cv_folds,
                                    scoring=scoring,
                                    use_log_transform=use_log_transform,
                                    save_as_version=selected_version
                                )
                            else:
                                tuning_results = AdminController.tune_model_hyperparameters(
                                    df, 
                                    method="grid", 
                                    param_grid=param_grid,
                                    cv=cv_folds,
                                    scoring=scoring,
                                    use_log_transform=use_log_transform,
                                    save_as_version=selected_version
                                )
                        else:
                            # Randomized search
                            param_distributions = {
                                'n_estimators': list(range(n_estimators_min, n_estimators_max + 1, 10)),
                                'max_depth': list(range(max_depth_min, max_depth_max + 1, 1)),
                                'min_samples_split': [2, 3, 5, 7, 10, 15],
                                'min_samples_leaf': [1, 2, 3, 4, 5],
                                'max_features': ['sqrt', 'log2', None]
                            }
                            
                            # Add None to max_depth if selected
                            if include_none_depth:
                                param_distributions['max_depth'].append(None)
                            
                            # Run randomized search with or without engineered features
                            if use_engineered_features:
                                tuning_results = AdminController.tune_model_with_engineered_features(
                                    df, 
                                    method="random", 
                                    param_distributions=param_distributions,
                                    n_iter=n_iter,
                                    cv=cv_folds,
                                    scoring=scoring,
                                    use_log_transform=use_log_transform,
                                    save_as_version=selected_version
                                )
                            else:
                                tuning_results = AdminController.tune_model_hyperparameters(
                                    df, 
                                    method="random", 
                                    param_distributions=param_distributions,
                                    n_iter=n_iter,
                                    cv=cv_folds,
                                    scoring=scoring,
                                    use_log_transform=use_log_transform,
                                    save_as_version=selected_version
                                )
                        
                        # Store results in session state
                        st.session_state.tuning_results = tuning_results
                        
                        # Rerun to display the results
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during hyperparameter tuning: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())


    with model_tabs[3]:  # Data Management tab
        st.subheader("Add New Data to Dataset")
        st.write("""
        Adding more data to your training dataset can significantly improve model accuracy.
        You can add individual records or upload a CSV file with multiple records.
        """)
        
        # Create tabs for different data addition methods
        data_tabs = st.tabs(["Add Single Record", "Upload CSV", "Generate Synthetic Data"])
        
        with data_tabs[0]:  # Add Single Record tab
            st.subheader("Add a Single Record")
            
            with st.form("add_record_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    years_exp = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
                    education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
                    job_role = st.selectbox("Job Role", [
                        "Front-end Developer", "Back-end Developer", "Full-stack Developer",
                        "Mobile Developer", "DevOps", "Data Scientist", "Embedded Engineer", "Game Developer"
                    ])
                
                with col2:
                    age = st.number_input("Age", min_value=18, max_value=80, value=30)
                    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                    salary = st.number_input("Monthly Salary (VND)", min_value=1000000, value=20000000, step=1000000)
                
                submit_button = st.form_submit_button("Add Record")
                
                if submit_button:
                    # Create a dictionary with the new record
                    new_record = {
                        'YearsExperience': years_exp,
                        'Education': education,
                        'JobRole': job_role,
                        'Age': age,
                        'Gender': gender,
                        'Salary': salary
                    }
                    
                    # Add the new record to the dataset
                    result = AdminController.add_new_data_to_dataset(new_record)
                    
                    if result['success']:
                        st.success(result['message'])
                        st.info("You may need to retrain your model to see the effects of the new data.")
                    else:
                        st.error(result['message'])
        
        with data_tabs[1]:  # Upload CSV tab
            st.subheader("Upload CSV File")
            st.write("""
            Upload a CSV file with new records to add to the dataset.
            The CSV should have the following columns: YearsExperience, Education, JobRole, Age, Gender, Salary
            """)
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    new_data_df = pd.read_csv(uploaded_file)
                    
                    # Display the uploaded data
                    st.write("Preview of uploaded data:")
                    st.dataframe(new_data_df.head())
                    
                    # Add option to append or replace data
                    data_action = st.radio(
                        "Choose how to use this data:",
                        ["Append to existing dataset", "Replace existing dataset"]
                    )
                    
                    # Add button to confirm action
                    if st.button("Process Data", key="process_csv_button"):
                        if data_action == "Append to existing dataset":
                            # Add the new data to the dataset
                            result = AdminController.add_new_data_to_dataset(new_data_df)
                            
                            if result['success']:
                                st.success(result['message'])
                                st.info("You may need to retrain your model to see the effects of the new data.")
                            else:
                                st.error(result['message'])
                        else:  # Replace existing dataset
                            # Replace the existing dataset with the new data
                            result = AdminController.replace_dataset(new_data_df)
                            
                            if result['success']:
                                st.success(result['message'])
                                st.warning("You must retrain your model to use this new dataset.")
                            else:
                                st.error(result['message'])
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
        
        with data_tabs[2]:  # Generate Synthetic Data tab
            st.subheader("Generate Synthetic Data")
            st.write("""
            Generate synthetic data based on the existing dataset to augment your training data.
            This can help improve model performance by providing more examples.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_samples = st.number_input("Number of Samples to Generate", min_value=1, max_value=1000, value=50)
            
            with col2:
                noise_level = st.slider("Noise Level", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                    help="Higher values create more diverse synthetic data")
            
            if st.button("Generate and Add Synthetic Data", key="generate_data_button"):
                # Call a function to generate synthetic data
                with st.spinner("Generating synthetic data..."):
                    result = AdminController.generate_synthetic_data(num_samples, noise_level)
                    
                    if result['success']:
                        # Display the generated data
                        st.write("Preview of generated data:")
                        st.dataframe(result['synthetic_data'].head())
                        
                        # Add button to confirm addition
                        if st.button("Add Generated Data to Dataset", key="add_synthetic_button"):
                            # Add the synthetic data to the dataset
                            add_result = AdminController.add_new_data_to_dataset(result['synthetic_data'])
                            
                            if add_result['success']:
                                st.success(add_result['message'])
                                st.info("You may need to retrain your model to see the effects of the new data.")
                            else:
                                st.error(add_result['message'])
                    else:
                        st.error(result['message'])