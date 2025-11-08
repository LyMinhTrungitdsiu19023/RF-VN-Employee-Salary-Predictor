import pandas as pd
from models.salary_model import SalaryPredictionModel

class UserController:
    """Controller for user-specific operations"""
    
    @staticmethod
    def get_model_versions():
        """
        Get list of available model versions
        
        Returns:
            list: List of version numbers
        """
        model = SalaryPredictionModel()
        return model.get_available_versions()
    
    @staticmethod
    def predict_salary(features, model_version=None):
        """
        Predict salary based on input features
        
        Args:
            features (dict): Dictionary with feature values
            model_version (int): Optional specific model version to use
            
        Returns:
            float: Predicted salary
        """
        model = SalaryPredictionModel(version=model_version)
        return model.predict_salary(features)
    
    @staticmethod
    def predict_batch(file, model_version=None):
        """
        Predict salaries for a batch of inputs from a CSV file
        
        Args:
            file: Uploaded CSV file
            model_version (int): Optional specific model version to use
            
        Returns:
            dict: Result with success status, message, and DataFrame with predictions
        """
        try:
            df = pd.read_csv(file)
            
            # Check required columns
            expected_cols = ['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender']
            if not all(col in df.columns for col in expected_cols):
                return {
                    "success": False,
                    "message": f"CSV must contain columns: {expected_cols}"
                }
            
            # Make predictions
            model = SalaryPredictionModel(version=model_version)
            result_df = model.predict_batch(df)
            
            return {
                "success": True,
                "message": "Predictions completed successfully",
                "data": result_df
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing CSV file: {e}"
            }