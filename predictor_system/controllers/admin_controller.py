import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from models.user import UserModel
from models.salary_model import SalaryPredictionModel
from config.settings import RATE_INR_TO_VND, CURRENCY_SYMBOL_VND
from utils.data_utils import load_data, save_data, append_data
from data.sample_data import create_sample_data
import os
import streamlit as st

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
    def get_employee_data():
        """
        Get employee data for visualization and model training
        
        Returns:
            DataFrame: Employee salary data
        """
        # Try to load from the CSV file
        df = load_data('employee_salary_data.csv')
        
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
            salary_series = df['Salary'] * RATE_INR_TO_VND if 'Salary' in df.columns else None
            sns.histplot(salary_series, kde=True, ax=ax)
            ax.set_title('Salary Distribution')
            ax.set_xlabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            ax.set_ylabel('Frequency')
            
        elif viz_type == "Salary by Experience":
            plot_df = df.copy()
            if 'Salary' in plot_df.columns:
                plot_df['Salary_VND'] = plot_df['Salary'] * RATE_INR_TO_VND
            y_col = 'Salary_VND' if 'Salary_VND' in plot_df.columns else 'Salary'
            sns.scatterplot(x='YearsExperience', y=y_col, data=plot_df, ax=ax)
            ax.set_title('Salary vs. Years of Experience')
            ax.set_xlabel('Years of Experience')
            ax.set_ylabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            
        elif viz_type == "Salary by Education":
            plot_df = df.copy()
            if 'Salary' in plot_df.columns:
                plot_df['Salary_VND'] = plot_df['Salary'] * RATE_INR_TO_VND
            sns.boxplot(x='Education', y='Salary_VND' if 'Salary_VND' in plot_df.columns else 'Salary', data=plot_df, ax=ax)
            ax.set_title('Salary by Education Level')
            ax.set_xlabel('Education Level')
            ax.set_ylabel(f'Salary ({CURRENCY_SYMBOL_VND})')
            plt.xticks(rotation=45)
            
        elif viz_type == "Salary by Job Role":
            plot_df = df.copy()
            if 'Salary' in plot_df.columns:
                plot_df['Salary_VND'] = plot_df['Salary'] * RATE_INR_TO_VND
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
        
        # Save the model with specific version if provided
        success = model.save_model(version=save_as_version)
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