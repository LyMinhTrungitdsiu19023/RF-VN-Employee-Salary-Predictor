import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models.user import UserModel
from models.salary_model import SalaryPredictionModel
from config.settings import CURRENCY_SYMBOL_VND
from utils.data_utils import load_data, save_data, append_data
from data.sample_data import create_sample_data
import os
import streamlit as st
import numpy as np

class AdminController:
    """Controller for admin-specific operations"""
    
    @staticmethod
    def get_all_users():
        """Get all users for the admin user management view"""
        return UserModel.get_all_users()
    
    @staticmethod
    def add_user(username, password, role, user_type=None):
        """
        Add a new user (mock implementation)
        
        Returns:
            dict: Result with success status and message
        """
        # In a real app, this would add to a database
        if not username or not password:
            return {"success": False, "message": "Username and password are required"}
            
        # Check if username already exists
        user_data = UserModel.get_all_users()
        if any(user["Username"] == username for user in user_data):
            return {"success": False, "message": f"Username '{username}' already exists"}
            
        # Create user data structure
        new_user_data = {
            "password": password,
            "role": role
        }
        if role == "User" and user_type:
            new_user_data["user_type"] = user_type
            
        return {
            "success": True, 
            "message": f"User '{username}' would be added in a real application.",
            "user_data": new_user_data
        }

    @staticmethod
    def replace_dataset(new_data_df, save_path=None):
        """
        Replace the existing dataset with new data
        
        Args:
            new_data_df (DataFrame): New data to replace existing dataset
            save_path (str, optional): Path to save the updated dataset
            
        Returns:
            dict: Result with success status and message
        """
        try:
            # Validate the new data
            print(f"Validating new dataset with shape: {new_data_df.shape}")
            
            # Check required columns
            required_columns = ['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']
            missing_columns = [col for col in required_columns if col not in new_data_df.columns]
            
            if missing_columns:
                return {
                    'success': False,
                    'message': f"Missing required columns in new data: {missing_columns}"
                }
            
            # Clean the data
            if 'Salary' in new_data_df.columns:
                new_data_df['Salary'] = new_data_df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
                new_data_df['Salary'] = pd.to_numeric(new_data_df['Salary'], errors='coerce')
            
            # Validate the data
            invalid_rows = new_data_df[new_data_df['Salary'].isna() | (new_data_df['Salary'] <= 0)].index.tolist()
            if invalid_rows:
                return {
                    'success': False,
                    'message': f"Invalid Salary values in rows: {invalid_rows}"
                }
            
            # Save the new dataset
            if save_path:
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save the new dataset
                new_data_df.to_csv(save_path, index=False)
                print(f"New dataset saved to {save_path}")
                
                return {
                    'success': True,
                    'message': f"Replaced existing dataset with {len(new_data_df)} records and saved to {save_path}",
                    'updated_df': new_data_df
                }
            else:
                # Use default path
                default_path = './data/processed_data/refined_salary_data.csv'
                
                # Create backup of existing file if it exists
                import os
                if os.path.exists(default_path):
                    import shutil
                    from datetime import datetime
                    backup_path = f'./data/processed_data/backup_salary_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                    shutil.copy2(default_path, backup_path)
                    print(f"Created backup of existing dataset at {backup_path}")
                
                # Save the new dataset
                new_data_df.to_csv(default_path, index=False)
                print(f"New dataset saved to {default_path}")
                
                return {
                    'success': True,
                    'message': f"Replaced existing dataset with {len(new_data_df)} records and saved to default location. A backup of the previous dataset was created.",
                    'updated_df': new_data_df
                }
            
        except Exception as e:
            print(f"Error replacing dataset: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'message': f"Error replacing dataset: {str(e)}"
            }
    
    @staticmethod
    def get_employee_data():
        """
        Get employee data for visualization and model training
        
        Returns:
            DataFrame: Employee salary data
        """
        # Try to load from the CSV file
        df = load_data('./data/processed_data/refined_salary_data.csv')
        
        if df is not None:
            return df
        else:
            # If file doesn't exist, show a warning and create sample data
            st.warning("employee_salary_data.csv not found. Using sample data for demonstration.")
            return create_sample_data(save_to_csv=False)  # Don't save the sample data
    
    @staticmethod
    def add_training_data(new_data):
        """
        Add new training data to the existing dataset
        
        Args:
            new_data (DataFrame): New data to add
            
        Returns:
            bool: True if successful, False otherwise
        """
        return append_data(new_data)
    
    @staticmethod
    def create_visualization(df, viz_type, x_var=None, y_var=None, plot_type=None):
        """
        Create visualizations for the admin data visualization view
        
        Args:
            df (DataFrame): Data to visualize
            viz_type (str): Type of visualization
            x_var, y_var (str, optional): Variables for custom plots
            plot_type (str, optional): Type of custom plot
            
        Returns:
            matplotlib.figure.Figure: The created visualization
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "Salary Distribution":
            # Convert salary to VND for plotting
            salary_series = df['Salary'] if 'Salary' in df.columns else None
            sns.histplot(salary_series, kde=True, ax=ax)
            ax.set_title('Salary Distribution')
            ax.set_xlabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            ax.set_ylabel('Frequency')
            
        elif viz_type == "Salary by Experience":
            plot_df = df.copy()

            y_col = 'Salary_VND' if 'Salary_VND' in plot_df.columns else 'Salary'
            sns.scatterplot(x='YearsExperience', y=y_col, data=plot_df, ax=ax)
            ax.set_title('Salary vs. Years of Experience')
            ax.set_xlabel('Years of Experience')
            ax.set_ylabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            
        elif viz_type == "Salary by Education":
            plot_df = df.copy()

            sns.boxplot(x='Education', y='Salary_VND' if 'Salary_VND' in plot_df.columns else 'Salary', data=plot_df, ax=ax)
            ax.set_title('Salary by Education Level')
            ax.set_xlabel('Education Level')
            ax.set_ylabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            plt.xticks(rotation=45)
            
        elif viz_type == "Salary by Job Role":
            plot_df = df.copy()

            sns.barplot(x='JobRole', y='Salary_VND' if 'Salary_VND' in plot_df.columns else 'Salary', data=plot_df, ax=ax)
            ax.set_title('Average Salary by Job Role')
            ax.set_xlabel('Job Role')
            ax.set_ylabel(f'Average Salary ({CURRENCY_SYMBOL_VND})')
            plt.xticks(rotation=45)
            
        elif viz_type == "Correlation Heatmap":
            # Create a copy of the dataframe with numeric columns only
            numeric_df = df.select_dtypes(include=['number'])  # Changed from pd.np.number to 'number'
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            
        elif viz_type == "Custom Plot" and x_var and y_var:
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
        
        return fig
        
    @staticmethod
    def get_available_versions():
        """Get list of available model versions"""
        model = SalaryPredictionModel()
        return model.get_available_versions()

    @staticmethod
    def get_model_parameters(version=None):
        """Get parameters for a specific model version"""
        model = SalaryPredictionModel()
        return model.get_model_parameters(version)

    @staticmethod
    def train_model(df, params, save_as_version=None):
        """
        Train a new salary prediction model
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            save_as_version (int, optional): Save as specific version number
            
        Returns:
            dict: Model performance metrics and new version number
        """
        # Remove Location column if it exists
        if 'Location' in df.columns:
            df = df.drop('Location', axis=1)
            
        model = SalaryPredictionModel()
        metrics = model.train_model(df, params)
        
        # Extract metrics for saving (exclude feature_importance as it's not serializable)
        save_metrics = {k: v for k, v in metrics.items() if k != 'feature_importance'}
        
        # Save the model with specific version if provided
        success = model.save_model(version=save_as_version, metrics=save_metrics)
        if success:
            version = model.version
            metrics['version'] = version
            metrics['success'] = True
            metrics['message'] = f"Model v{version} {'updated' if save_as_version else 'created'} successfully"
        else:
            metrics['success'] = False
            metrics['message'] = "Failed to save model"
            
        return metrics

    @staticmethod
    def delete_model_version(version):
        """
        Delete a specific model version
        
        Args:
            version (int): Version number to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        model = SalaryPredictionModel()
        return model.delete_version(version)
    
    @staticmethod
    def get_model_info(version=None):
        """
        Get information about available model versions
        
        Args:
            version (int, optional): Specific version to get info for
            
        Returns:
            dict: Model version information
        """
        model = SalaryPredictionModel(version=version)
        versions = model.get_available_versions()
        
        info = {
            'versions': versions,
            'latest': max(versions) if versions else None,
            'count': len(versions)
        }
        
        if version and version in versions:
            info['selected'] = version
            try:
                metadata = joblib.load(os.path.join(model.model_dir, f'v{version}', 'metadata.pkl'))
                info['metadata'] = metadata
            except:
                info['metadata'] = None
                
        return info

    @staticmethod
    def tune_model_hyperparameters(df):
        model = SalaryPredictionModel()
        tuning_results = model.randomized_hyperparameter_search(df, n_iter=20)
        
        # Save the tuned model and results
        model.save_tuning_results(tuning_results)
        
        return {
            'success': True,
            'best_params': tuning_results['best_params'],
            'best_score': tuning_results['best_score'],
            'version': model.version
        }
    

    @staticmethod
    def train_model_with_log_transform(df, params, save_as_version=None):
        """
        Train a salary prediction model with log-transformed target
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            save_as_version (int, optional): Save as specific version number
            
        Returns:
            dict: Model performance metrics and new version number
        """
        # Remove Location column if it exists
        if 'Location' in df.columns:
            df = df.drop('Location', axis=1)
            
        model = SalaryPredictionModel()
        metrics = model.train_model_with_log_transform(df, params, test_size=params.get('test_size', 0.2))
        
        # Extract metrics for saving (exclude feature_importance as it's not serializable)
        save_metrics = {k: v for k, v in metrics.items() if k != 'feature_importance'}
        
        # Save the model with specific version if provided
        success = model.save_model(version=save_as_version, metrics=save_metrics)
        if success:
            version = model.version
            metrics['version'] = version
            metrics['success'] = True
            metrics['message'] = f"Model v{version} {'updated' if save_as_version else 'created'} successfully"
        else:
            metrics['success'] = False
            metrics['message'] = "Failed to save model"
            
        return metrics

    @staticmethod
    def tune_model_hyperparameters(df, method="random", param_grid=None, param_distributions=None, 
                                cv=5, scoring='neg_mean_squared_error', n_iter=20, 
                                use_log_transform=True, save_as_version=None):
        """
        Tune model hyperparameters using grid search or randomized search
        
        Args:
            df (DataFrame): Training data
            method (str): "grid" for GridSearchCV or "random" for RandomizedSearchCV
            param_grid (dict): Parameter grid for grid search
            param_distributions (dict): Parameter distributions for randomized search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for randomized search
            use_log_transform (bool): Whether to use log transformation for target
            save_as_version (int, optional): Save as specific version number
            
        Returns:
            dict: Tuning results including best parameters and score
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Remove Location column if it exists
        if 'Location' in df.columns:
            df = df.drop('Location', axis=1)
            
        model = SalaryPredictionModel()
        
        # Run hyperparameter tuning
        if method == "grid":
            tuning_results = model.tune_hyperparameters(
                df, 
                param_grid=param_grid, 
                cv=cv, 
                scoring=scoring
            )
        else:  # randomized search
            tuning_results = model.randomized_hyperparameter_search(
                df, 
                param_distributions=param_distributions, 
                n_iter=n_iter, 
                cv=cv, 
                scoring=scoring
            )
        
        # Create visualization of CV results
        cv_results = tuning_results['cv_results']
        
        # Create a figure for CV results visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the top 10 results
        if isinstance(cv_results, pd.DataFrame):
            top_results = cv_results.sort_values(by='mean_test_score', ascending=False).head(10)
            
            # Create a bar chart of the top results
            sns.barplot(x='rank_test_score', y='mean_test_score', data=top_results, ax=ax)
            ax.set_title('Top 10 Hyperparameter Combinations')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Mean Test Score')
            
            # Add the figure to the results
            tuning_results['cv_results_fig'] = fig
        
        # Save the tuned model with metrics
        best_params = tuning_results['best_params']
        best_score = tuning_results['best_score']
        
        # Create metrics dict
        metrics = {
            'best_score': best_score,
            'tuning_method': method,
            'cv_folds': cv,
            'scoring': scoring
        }
        
        # Save the model with specific version if provided
        success = model.save_model(version=save_as_version, metrics=metrics)
        
        if success:
            # Save tuning results separately
            model.save_tuning_results(tuning_results)
            
            version = model.version
            return {
                'success': True,
                'version': version,
                'best_params': best_params,
                'best_score': best_score,
                'cv_results_fig': fig if 'cv_results_fig' in tuning_results else None
            }
        else:
            return {
                'success': False,
                'message': "Failed to save model"
            }
        
    @staticmethod
    def train_model_with_engineered_features(df, params, save_as_version=None, use_log_transform=True):
        """
        Train a Random Forest model with engineered features for better prediction accuracy
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            save_as_version (int, optional): Save as specific version number
            use_log_transform (bool): Whether to use log transformation for target
            
        Returns:
            dict: Model performance metrics and new version number
        """
        try:
            # Debug info
            print("Starting train_model_with_engineered_features")
            print(f"DataFrame shape: {df.shape}")
            print(f"Parameters: {params}")
            print(f"use_log_transform: {use_log_transform}")
            
            # Remove Location column if it exists
            if 'Location' in df.columns:
                df = df.drop('Location', axis=1)
                print("Dropped Location column")
                    
            model = SalaryPredictionModel()
            
            # Check if the required columns exist in the dataframe
            required_columns = ['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Pass test_size as part of params instead of as a separate parameter
            print("Calling model.train_model_with_engineered_features")
            metrics = model.train_model_with_engineered_features(
                df, 
                params, 
                test_size=params.get('test_size', 0.2), 
                use_log_transform=use_log_transform
            )
            
            print(f"Training completed. Metrics: {metrics.keys()}")
            
            # Extract metrics for saving (exclude feature_importance as it's not serializable)
            save_metrics = {
                'mse': metrics.get('mse', 0),
                'rmse': metrics.get('rmse', 0),
                'r2': metrics.get('r2', 0),
                'mae': metrics.get('mae', 0),
                'mape': metrics.get('mape', 0),
                'log_transformed': use_log_transform,
                'engineered_features': True
            }
            
            # Save the model with specific version if provided
            success = model.save_model(version=save_as_version, metrics=save_metrics)
            
            if success:
                version = model.version
                metrics['version'] = version
                metrics['success'] = True
                metrics['message'] = f"Random Forest model v{version} with engineered features {'updated' if save_as_version else 'created'} successfully"
                
                # Add OOB score to message if available
                if metrics.get('oob_score') is not None:
                    metrics['message'] += f" (OOB Score: {metrics['oob_score']:.4f})"
                    
                print(f"Model saved successfully as version {version}")
            else:
                metrics['success'] = False
                metrics['message'] = "Failed to save model"
                print("Failed to save model")
                    
            return metrics
        except Exception as e:
            # Log the error
            print(f"Error in train_model_with_engineered_features: {e}")
            import traceback
            traceback.print_exc()
            
            # Return a failure metrics object
            return {
                'success': False,
                'message': f"Error training model: {str(e)}",
                'rmse': 0,
                'r2': 0,
                'mae': 0,
                'mape': 0
            }

    @staticmethod
    def tune_model_with_engineered_features(df, method="random", param_grid=None, param_distributions=None, 
                                        cv=5, scoring='neg_mean_squared_error', n_iter=20, 
                                        use_log_transform=True, save_as_version=None):
        """
        Tune model hyperparameters using engineered features
        
        Args:
            df (DataFrame): Training data
            method (str): "grid" for GridSearchCV or "random" for RandomizedSearchCV
            param_grid (dict): Parameter grid for grid search
            param_distributions (dict): Parameter distributions for randomized search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for randomized search
            use_log_transform (bool): Whether to use log transformation for target
            save_as_version (int, optional): Save as specific version number
            
        Returns:
            dict: Tuning results including best parameters and score
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np
        from sklearn.preprocessing import LabelEncoder
        
        try:
            # Remove Location column if it exists
            if 'Location' in df.columns:
                df = df.drop('Location', axis=1)
            
            # Ensure Salary column is numeric
            df_copy = df.copy()
            if 'Salary' in df_copy.columns:
                # Convert Salary to numeric, handling any non-numeric values
                df_copy['Salary'] = pd.to_numeric(df_copy['Salary'].astype(str).str.replace(',', '').str.replace('"', ''), errors='coerce')
                # Drop rows with invalid Salary values
                df_copy = df_copy.dropna(subset=['Salary'])
                # Ensure Salary is positive
                df_copy = df_copy[df_copy['Salary'] > 0]
            
            # Create a new SalaryPredictionModel instance
            model = SalaryPredictionModel()
            
            # Instead of trying to split the data and do our own feature engineering,
            # let's use the model's built-in methods for hyperparameter tuning
            
            # Run hyperparameter tuning with engineered features
            if method == "grid":
                tuning_results = model.tune_hyperparameters_with_engineered_features(
                    df_copy,  # Use the cleaned dataframe
                    param_grid=param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    use_log_transform=use_log_transform
                )
            else:  # randomized search
                tuning_results = model.randomized_hyperparameter_search(
                    df_copy,  # Use the cleaned dataframe
                    param_distributions=param_distributions, 
                    n_iter=n_iter, 
                    cv=cv, 
                    scoring=scoring
                )
            
            # Get the best model
            best_model = tuning_results.get('model')
            best_params = tuning_results.get('best_params', {})
            best_score = tuning_results.get('best_score', 0)
            
            # Get metrics if available in tuning_results
            mse = tuning_results.get('mse', 0)
            rmse = tuning_results.get('rmse', 0)
            r2 = tuning_results.get('r2', 0)
            mae = tuning_results.get('mae', 0)
            mape = tuning_results.get('mape', 0)
            
            # Create visualization of CV results
            cv_results = tuning_results.get('cv_results')
            
            # Create a figure for CV results visualization
            fig = None
            if cv_results is not None and isinstance(cv_results, pd.DataFrame):
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get the top 10 results
                top_results = cv_results.sort_values(by='mean_test_score', ascending=False).head(10)
                
                # Create a bar chart of the top results
                sns.barplot(x='rank_test_score', y='mean_test_score', data=top_results, ax=ax)
                ax.set_title('Top 10 Hyperparameter Combinations')
                ax.set_xlabel('Rank')
                ax.set_ylabel('Mean Test Score')
            
            # Create metrics dict
            metrics = {
                'best_score': best_score,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'tuning_method': method,
                'cv_folds': cv,
                'scoring': scoring,
                'engineered_features': True
            }
            
            # Save the model with specific version if provided
            success = model.save_model(version=save_as_version, metrics=metrics)
            
            if success:
                # Save tuning results separately
                model.save_tuning_results(tuning_results)
                
                version = model.version
                return {
                    'success': True,
                    'version': version,
                    'best_params': best_params,
                    'best_score': best_score,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'mape': mape,
                    'cv_results_fig': fig,
                    'engineered_features': True
                }
            else:
                return {
                    'success': False,
                    'message': "Failed to save model"
                }
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in tune_model_with_engineered_features: {e}")
            print(error_details)
            
            return {
                'success': False,
                'message': f"Error during hyperparameter tuning: {str(e)}",
                'error_details': error_details
            }
        
    @staticmethod
    def add_new_data_to_dataset(new_data, save_path=None):
        """
        Add new data to the existing dataset
        
        Args:
            new_data (DataFrame or dict): New data to add
            save_path (str, optional): Path to save the updated dataset
            
        Returns:
            dict: Result with success status and message
        """
        try:
            # Get current dataset
            current_df = AdminController.get_employee_data()
            print(f"Current dataset shape: {current_df.shape}")
            
            # Convert dict to DataFrame if needed
            if isinstance(new_data, dict):
                new_data_df = pd.DataFrame([new_data])
            elif isinstance(new_data, pd.DataFrame):
                new_data_df = new_data
            else:
                return {
                    'success': False,
                    'message': "Invalid data format. Please provide a DataFrame or dictionary."
                }
            
            print(f"New data shape: {new_data_df.shape}")
            
            # Ensure new data has the same columns as the current dataset
            required_columns = ['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']
            missing_columns = [col for col in required_columns if col not in new_data_df.columns]
            
            if missing_columns:
                return {
                    'success': False,
                    'message': f"Missing required columns in new data: {missing_columns}"
                }
            
            # Clean the new data
            if 'Salary' in new_data_df.columns:
                new_data_df['Salary'] = new_data_df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
                new_data_df['Salary'] = pd.to_numeric(new_data_df['Salary'], errors='coerce')
            
            # Validate the new data
            invalid_rows = new_data_df[new_data_df['Salary'].isna() | (new_data_df['Salary'] <= 0)].index.tolist()
            if invalid_rows:
                return {
                    'success': False,
                    'message': f"Invalid Salary values in rows: {invalid_rows}"
                }
            
            # Append the new data to the current dataset
            updated_df = pd.concat([current_df, new_data_df], ignore_index=True)
            print(f"Updated dataset shape: {updated_df.shape}")
            
            # Save the updated dataset if a path is provided
            if save_path:
                # Ensure directory exists
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save the updated dataset
                updated_df.to_csv(save_path, index=False)
                print(f"Updated dataset saved to {save_path}")
                
                return {
                    'success': True,
                    'message': f"Added {len(new_data_df)} new records to the dataset and saved to {save_path}",
                    'updated_df': updated_df
                }
            else:
                # Use default path
                default_path = './data/processed_data/processed_salary_data_standardized.csv'
                updated_df.to_csv(default_path, index=False)
                print(f"Updated dataset saved to {default_path}")
                
                return {
                    'success': True,
                    'message': f"Added {len(new_data_df)} new records to the dataset and saved to default location",
                    'updated_df': updated_df
                }
                
        except Exception as e:
            print(f"Error adding new data to dataset: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'message': f"Error adding new data: {str(e)}"
            }
        
    @staticmethod
    def generate_synthetic_data(num_samples=50, noise_level=0.1):
        """
        Generate synthetic data based on the existing dataset
        
        Args:
            num_samples (int): Number of synthetic samples to generate
            noise_level (float): Level of noise to add (0.0 to 1.0)
            
        Returns:
            dict: Result with success status, message, and synthetic data
        """
        try:
            # Get current dataset
            current_df = AdminController.get_employee_data()
            print(f"Current dataset shape: {current_df.shape}")
            
            # Create a copy to avoid modifying the original
            df = current_df.copy()
            
            # Convert categorical columns to numerical for easier sampling
            categorical_cols = ['Education', 'JobRole', 'Gender']
            for col in categorical_cols:
                df[f'{col}_Code'] = pd.factorize(df[col])[0]
            
            # Get the numerical columns for sampling
            numerical_cols = ['YearsExperience', 'Age', 'Salary'] + [f'{col}_Code' for col in categorical_cols]
            
            # Calculate mean and standard deviation for each numerical column
            means = df[numerical_cols].mean()
            stds = df[numerical_cols].std()
            
            # Generate synthetic data
            synthetic_data = pd.DataFrame()
            
            for col in numerical_cols:
                # Generate random values around the mean with controlled noise
                synthetic_values = np.random.normal(
                    loc=means[col],
                    scale=stds[col] * (1 + noise_level),
                    size=num_samples
                )
                
                # For categorical codes, round to nearest integer
                if col.endswith('_Code'):
                    synthetic_values = np.round(synthetic_values).clip(0, df[col].max())
                    
                synthetic_data[col] = synthetic_values
            
            # Convert categorical codes back to categories
            for col in categorical_cols:
                # Get the mapping from code to category
                code_to_category = dict(enumerate(df[col].unique()))
                
                # Map the codes to categories
                synthetic_data[col] = synthetic_data[f'{col}_Code'].map(
                    lambda x: code_to_category.get(int(x % len(code_to_category)), code_to_category[0])
                )
                
                # Drop the code column
                synthetic_data.drop(f'{col}_Code', axis=1, inplace=True)
            
            # Ensure YearsExperience is non-negative and reasonable
            synthetic_data['YearsExperience'] = synthetic_data['YearsExperience'].clip(0, 40).round(0)
            
            # Ensure Age is reasonable
            synthetic_data['Age'] = synthetic_data['Age'].clip(18, 70).round(0)
            
            # Ensure Salary is non-negative and reasonable
            synthetic_data['Salary'] = synthetic_data['Salary'].clip(5000000, 500000000).round(-5)  # Round to nearest 100,000
            
            print(f"Generated {num_samples} synthetic samples")
            
            return {
                'success': True,
                'message': f"Successfully generated {num_samples} synthetic samples",
                'synthetic_data': synthetic_data
            }
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'message': f"Error generating synthetic data: {str(e)}"
            }