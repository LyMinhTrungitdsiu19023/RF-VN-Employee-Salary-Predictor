import os
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

class SalaryPredictionModel:
    """Model for salary prediction with versioning support"""
    
    def __init__(self, version=None):
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.is_loaded = False
        self.version = version
        self.model_dir = 'model_versions'
        
        # Ensure model versions directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Try to load the model
        try:
            if version:
                self._load_specific_version(version)
            else:
                self._load_latest_version()
        except Exception:
            # Model not loaded, will use mock predictions
            pass
            
    def _load_specific_version(self, version):
        """Load a specific version of the model"""
        version_path = os.path.join(self.model_dir, f'v{version}')
        self.model = joblib.load(os.path.join(version_path, 'salary_model.pkl'))
        self.scaler = joblib.load(os.path.join(version_path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(version_path, 'label_encoders.pkl'))
        self.version = version
        self.is_loaded = True
    
    def _load_latest_version(self):
        """Load the latest version of the model"""
        versions = self.get_available_versions()
        if versions:
            latest_version = max(versions)
            self._load_specific_version(latest_version)
    
    def get_available_versions(self):
        """Get list of available model versions"""
        try:
            versions = []
            for dirname in os.listdir(self.model_dir):
                if dirname.startswith('v') and os.path.isdir(os.path.join(self.model_dir, dirname)):
                    try:
                        version_num = int(dirname[1:])
                        versions.append(version_num)
                    except ValueError:
                        continue
            return sorted(versions)
        except FileNotFoundError:
            return []
            
    def get_model_parameters(self, version=None):
        """Get model parameters and metadata for a specific version"""
        if version is None:
            version = self.version
            
        if version is None:
            return None
            
        try:
            version_path = os.path.join(self.model_dir, f'v{version}')
            
            # Load model first as it's essential
            model = joblib.load(os.path.join(version_path, 'salary_model.pkl'))
            
            # Get basic model parameters
            params = {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'max_features': model.max_features,
                'random_state': model.random_state
            }
            
            # Try to load metadata, but don't fail if it doesn't exist
            try:
                metadata = joblib.load(os.path.join(version_path, 'metadata.pkl'))
                params.update({
                    'version': version,
                    'created_at': metadata.get('created_at'),
                    'feature_names': metadata.get('feature_names')
                })
            except Exception:
                params['version'] = version
            
            return params
            
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return None
    
    def predict_salary(self, features):
        """
        Predict salary based on input features
        
        Args:
            features (dict): Dictionary with feature values
            
        Returns:
            float: Predicted salary
        """
        if self.is_loaded:
            # Encode categorical features
            input_dict = {
                'YearsExperience': features['experience'],
                'Education': self.label_encoders['Education'].transform([features['education']])[0],
                'JobRole': self.label_encoders['JobRole'].transform([features['job_role']])[0],
                'Age': features['age'],
                'Gender': self.label_encoders['Gender'].transform([features['gender']])[0],
            }
            input_df = pd.DataFrame([input_dict])
            input_scaled = self.scaler.transform(input_df)
            return self.model.predict(input_scaled)[0]
        else:
            # Mock prediction for demo
            return self._mock_predict(features)
    
    def predict_batch(self, df):
        """
        Predict salaries for a batch of inputs
        
        Args:
            df (DataFrame): DataFrame with feature columns
            
        Returns:
            DataFrame: Original DataFrame with predictions added
        """
        df_copy = df.copy()
        
        if self.is_loaded:
            # Encode categorical columns
            for col in ['Education', 'JobRole', 'Gender']:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col])
            
            X = df_copy[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender']]
            X_scaled = self.scaler.transform(X)
            df_copy['PredictedSalary'] = self.model.predict(X_scaled).astype(int)
        else:
            # Mock batch predictions
            mock_salaries = []
            for _, row in df_copy.iterrows():
                features = {
                    'experience': row['YearsExperience'],
                    'education': row['Education'],
                    'job_role': row['JobRole'],
                    'age': row['Age'],
                    'gender': row['Gender']
                }
                mock_salaries.append(int(self._mock_predict(features)))
            
            df_copy['PredictedSalary'] = mock_salaries
        
        return df_copy
    
    def train_model(self, df, params, test_size=0.2):
        """
        Train a new salary prediction model
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            test_size (float): Test split ratio
            
        Returns:
            dict: Model performance metrics
        """
        # Remove Location column if it exists
        if 'Location' in df.columns:
            df = df.drop('Location', axis=1)
            
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
            X, y, test_size=test_size, random_state=params['random_state']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fix max_features parameter
        max_features = params['max_features']
        if max_features == 'auto':
            max_features = 'sqrt'  # 'auto' is deprecated, use 'sqrt' instead
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features,  # Using the fixed parameter
            random_state=params['random_state']
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Save model artifacts
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.is_loaded = True
        
        # Return metrics and feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'feature_importance': feature_importance
        }
    
    def delete_version(self, version):
        """Delete a specific model version"""
        try:
            version_path = os.path.join(self.model_dir, f'v{version}')
            if os.path.exists(version_path):
                # List of files to delete
                files_to_delete = ['salary_model.pkl', 'scaler.pkl', 
                                 'label_encoders.pkl', 'metadata.pkl']
                
                # Delete each file if it exists
                for file in files_to_delete:
                    file_path = os.path.join(version_path, file)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"Error deleting {file}: {e}")
                
                # Delete any remaining files
                for file in os.listdir(version_path):
                    try:
                        os.remove(os.path.join(version_path, file))
                    except Exception as e:
                        print(f"Error deleting additional file {file}: {e}")
                
                # Remove the directory
                try:
                    os.rmdir(version_path)
                except Exception as e:
                    print(f"Error removing directory: {e}")
                    return False
                
                # If this was our current version, reset the model
                if self.version == version:
                    self.model = None
                    self.scaler = None
                    self.label_encoders = None
                    self.is_loaded = False
                    self.version = None
                    
                return True
            return False
        except Exception as e:
            print(f"Error deleting model version: {e}")
            return False

    def save_model(self, version=None):
        """Save model artifacts as a new version or update existing version"""
        if self.is_loaded:
            if version is None:
                # Get next version number for new version
                versions = self.get_available_versions()
                version = 1 if not versions else max(versions) + 1
            
            # Create or use existing version directory
            version_path = os.path.join(self.model_dir, f'v{version}')
            os.makedirs(version_path, exist_ok=True)
            
            # Save model files
            joblib.dump(self.model, os.path.join(version_path, 'salary_model.pkl'))
            joblib.dump(self.scaler, os.path.join(version_path, 'scaler.pkl'))
            joblib.dump(self.label_encoders, os.path.join(version_path, 'label_encoders.pkl'))
            
            # Save metadata
            metadata = {
                'version': version,
                'created_at': datetime.datetime.now().isoformat(),
                'feature_names': list(self.label_encoders.keys())
            }
            joblib.dump(metadata, os.path.join(version_path, 'metadata.pkl'))
            
            self.version = version
            return True
        return False
    
    def _mock_predict(self, features):
        """Generate mock predictions for demo purposes"""
        base_salary = 50000
        exp_factor = features['experience'] * 5000
        
        edu_factor = 0
        if features['education'] == 'Bachelor':
            edu_factor = 20000
        elif features['education'] == 'Master':
            edu_factor = 40000
        elif features['education'] == 'PhD':
            edu_factor = 60000
        
        role_factor = 0
        if features['job_role'] == 'Data Scientist':
            role_factor = 30000
        elif features['job_role'] == 'Software Engineer':
            role_factor = 25000
        elif features['job_role'] == 'Manager':
            role_factor = 35000
        elif features['job_role'] == 'Analyst':
            role_factor = 15000
        elif features['job_role'] == 'HR':
            role_factor = 10000
        
        age_factor = (features['age'] - 22) * 1000
        
        # Add some random noise
        import random
        noise = random.uniform(-10000, 10000)
        
        # Calculate mock salary (removed location factor)
        mock_salary = base_salary + exp_factor + edu_factor + role_factor + age_factor + noise
        return mock_salary