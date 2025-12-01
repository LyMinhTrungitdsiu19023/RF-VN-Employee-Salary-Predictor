import os
import joblib
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class SalaryPredictionModel:
    """Model for predicting software engineer salaries in Vietnam"""
    
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
        except Exception as e:
            print(f"Could not load model: {e}")
            # Model not loaded, will use mock predictions
            pass
            
    def _load_specific_version(self, version):
        """Load a specific version of the model"""
        version_path = os.path.join(self.model_dir, f'v{version}')
        self.model = joblib.load(os.path.join(version_path, 'salary_model.pkl'))
        self.scaler = joblib.load(os.path.join(version_path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(version_path, 'label_encoders.pkl'))
        
        # Load metadata to get additional information
        try:
            metadata = joblib.load(os.path.join(version_path, 'metadata.pkl'))
            if 'is_log_transformed' in metadata:
                self.is_log_transformed = metadata['is_log_transformed']
            if 'uses_engineered_features' in metadata:
                self.uses_engineered_features = metadata['uses_engineered_features']
            if 'engineered_feature_names' in metadata:
                self.feature_names = metadata['engineered_feature_names']
        except Exception as e:
            print(f"Warning: Could not load additional metadata: {e}")
        
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
    
    # def predict_salary(self, features):
    #     """
    #     Predict salary based on input features
        
    #     Args:
    #         features (dict): Dictionary with feature values
            
    #     Returns:
    #         float: Predicted salary in VND
    #     """
    #     if self.is_loaded:
    #         # Encode categorical features
    #         input_dict = {
    #             'YearsExperience': features['experience'],
    #             'Education': self.label_encoders['Education'].transform([features['education']])[0],
    #             'JobRole': self.label_encoders['JobRole'].transform([features['job_role']])[0],
    #             'Age': features['age'],
    #             'Gender': self.label_encoders['Gender'].transform([features['gender']])[0],
    #         }
    #         input_df = pd.DataFrame([input_dict])
    #         input_scaled = self.scaler.transform(input_df)
    #         return self.model.predict(input_scaled)[0]
    #     else:
    #         # Mock prediction for demo
    #         return self._mock_predict(features)

    def predict_salary(self, features):
        """
        Predict salary based on input features
        
        Args:
            features (dict): Dictionary with feature values
            
        Returns:
            float: Predicted salary in VND
        """
        if self.is_loaded:
            # Check if we need to use engineered features
            if hasattr(self, 'uses_engineered_features') and self.uses_engineered_features:
                return self.predict_salary_with_engineered_features(features)
            
            # Regular prediction without engineered features
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
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # If model was trained with log-transformed targets, transform back
            if hasattr(self, 'is_log_transformed') and self.is_log_transformed:
                prediction = np.expm1(prediction)
                
            return prediction
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
            # Check if we need to use engineered features
            if hasattr(self, 'uses_engineered_features') and self.uses_engineered_features:
                # For models with engineered features, we need to:
                # 1. Engineer features for the entire batch
                # 2. Encode categorical features
                # 3. Apply scaling
                # 4. Make predictions
                
                # Engineer features
                try:
                    df_engineered = self.engineer_features(df_copy)
                    
                    # Encode categorical features
                    for col in df_engineered.select_dtypes(include=['object']).columns:
                        if col in self.label_encoders:
                            df_engineered[col] = self.label_encoders[col].transform(df_engineered[col])
                    
                    # Ensure all required columns are present
                    if hasattr(self, 'feature_names'):
                        missing_cols = set(self.feature_names) - set(df_engineered.columns)
                        for col in missing_cols:
                            df_engineered[col] = 0  # Default value for missing columns
                        
                        # Select only the columns used during training
                        X = df_engineered[self.feature_names]
                    else:
                        # If feature names are not available, use all columns except Salary
                        X = df_engineered.drop('Salary', axis=1, errors='ignore')
                    
                    # Scale features
                    X_scaled = self.scaler.transform(X)
                    
                    # Make predictions
                    predictions = self.model.predict(X_scaled)
                    
                    # Transform back if log transformed
                    if hasattr(self, 'is_log_transformed') and self.is_log_transformed:
                        predictions = np.expm1(predictions)
                    
                    df_copy['PredictedSalary'] = predictions.astype(int)
                    
                except Exception as e:
                    print(f"Error in predict_batch with engineered features: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Fall back to individual predictions if batch processing fails
                    predictions = []
                    for _, row in df_copy.iterrows():
                        features = {
                            'experience': row['YearsExperience'],
                            'education': row['Education'],
                            'job_role': row['JobRole'],
                            'age': row['Age'],
                            'gender': row['Gender']
                        }
                        predictions.append(self.predict_salary(features))
                    
                    df_copy['PredictedSalary'] = predictions
            else:
                # Standard model without engineered features
                # Encode categorical columns
                for col in ['Education', 'JobRole', 'Gender']:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col])
                
                X = df_copy[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender']]
                X_scaled = self.scaler.transform(X)
                
                # Make predictions
                predictions = self.model.predict(X_scaled)
                
                # Transform back if log transformed
                if hasattr(self, 'is_log_transformed') and self.is_log_transformed:
                    predictions = np.expm1(predictions)
                    
                df_copy['PredictedSalary'] = predictions.astype(int)
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
        Train a new salary prediction model for Vietnam software engineers
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            test_size (float): Test split ratio
            
        Returns:
            dict: Model performance metrics
        """
        # Clean the data
        # Convert CompTotal from string to numeric if it exists
        if 'Salary' in df.columns:
            df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
            # Rename to Salary for consistency
            #         
        # Filter out rows with invalid salary values
        df = df[df['Salary'] > 0]
        
        # Fill missing values in YearsExperience
        if 'YearsExperience' in df.columns:
            df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
        
        # Select only the columns we need
        if set(['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']).issubset(df.columns):
            df = df[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']]
        
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
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
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
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'feature_importance': feature_importance
        }
        
        return metrics
    
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

    def train_and_save(self, X_train, y_train, X_test, y_test, version=None):
        """
        Train the model and save it with evaluation metrics
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Testing data
            version: Optional version number
            
        Returns:
            dict: Training results including metrics and version number
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'mape': float(mape)
        }
        
        # Save the model with metrics
        saved_version = self.save_model(version, metrics)
        
        return {
            'success': True,
            'version': saved_version,
            'metrics': metrics
        }
    
    def save_model(self, version=None, metrics=None):
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
                'feature_names': list(self.label_encoders.keys()),
                'model_type': 'Vietnam Software Engineer Salary Prediction'
            }
            
            # Add information about log transformation
            if hasattr(self, 'is_log_transformed'):
                metadata['is_log_transformed'] = self.is_log_transformed
                
            # Add information about engineered features
            if hasattr(self, 'uses_engineered_features'):
                metadata['uses_engineered_features'] = self.uses_engineered_features
                if hasattr(self, 'feature_names'):
                    metadata['engineered_feature_names'] = self.feature_names
            
            # Add metrics to metadata if provided
            if metrics:
                metadata['metrics'] = metrics
            
            joblib.dump(metadata, os.path.join(version_path, 'metadata.pkl'))
            
            self.version = version
            return True
        return False
    
    def _mock_predict(self, features):
        """Generate mock predictions for Vietnam software engineer salaries"""
        # Base salary in VND (around $2000 USD)
        base_salary = 20000000
        
        # Experience has a strong impact
        exp_factor = features['experience'] * 2000000
        
        # Education factor
        edu_factor = 0
        if features['education'] == 'High School':
            edu_factor = 0
        elif features['education'] == 'Bachelor':
            edu_factor = 5000000
        elif features['education'] == 'Master':
            edu_factor = 10000000
        elif features['education'] == 'PhD':
            edu_factor = 20000000
        
        # Job role factor
        role_factor = 0
        if features['job_role'] == 'Data Scientist':
            role_factor = 8000000
        elif features['job_role'] == 'Back-end Developer':
            role_factor = 5000000
        elif features['job_role'] == 'Front-end Developer':
            role_factor = 4000000
        elif features['job_role'] == 'Mobile Developer':
            role_factor = 5500000
        elif features['job_role'] == 'Embedded Engineer':
            role_factor = 6000000
        elif features['job_role'] == 'DevOps':
            role_factor = 7000000
        elif features['job_role'] == 'Full-stack Developer':
            role_factor = 6500000
        elif features['job_role'] == 'Game Developer':
            role_factor = 7500000
        
        # Age factor (small impact)
        age_factor = (features['age'] - 20) * 200000
        
        # Add some random noise
        import random
        noise = random.uniform(-2000000, 2000000)
        
        # Calculate mock salary
        mock_salary = base_salary + exp_factor + edu_factor + role_factor + age_factor + noise
        return max(mock_salary, 5000000)  # Ensure minimum wage
    
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
                    'feature_names': metadata.get('feature_names'),
                    'metrics': metadata.get('metrics'),  # Ensure metrics are included
                    'uses_engineered_features': metadata.get('uses_engineered_features', False),
                    'is_log_transformed': metadata.get('is_log_transformed', False)
                })
            except Exception:
                params['version'] = version
            
            return params
            
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            return None
        
    def tune_hyperparameters(self, df, param_grid=None, cv=5, scoring='neg_mean_squared_error'):
        """
        Tune model hyperparameters using GridSearchCV
        
        Args:
            df (DataFrame): Training data
            param_grid (dict): Grid of parameters to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            
        Returns:
            dict: Best parameters and cross-validation results
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        # Clean the data
        if 'Salary' in df.columns:
            df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                
        # Filter out rows with invalid salary values
        df = df[df['Salary'] > 0]
        
        # Fill missing values in YearsExperience
        if 'YearsExperience' in df.columns:
            df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
        
        # Select only the columns we need
        if set(['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']).issubset(df.columns):
            df = df[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']]
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        
        # Make sure random_state is only specified once
        # Remove random_state from param_grid if it exists
        if 'random_state' in param_grid:
            del param_grid['random_state']
        
        # Create base model with fixed random_state
        rf = RandomForestRegressor(random_state=42)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # Use all available cores
            verbose=1,
            return_train_score=True
        )
        
        # Fit the grid search
        grid_search.fit(X_scaled, y)
        
        # Get best parameters and score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Add random_state to best_params
        best_params['random_state'] = 42
        
        # Create a model with the best parameters
        best_model = RandomForestRegressor(**best_params)
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model on the training set
        best_model.fit(X_train_scaled, y_train)
        
        # Make predictions on test set
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        med_ae = np.median(np.abs(y_test - y_pred))
        med_ape = np.median(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Save the tuned model
        self.model = best_model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.is_loaded = True
        
        # Fix: Remove reference to undefined variable use_log_transform
        # Instead, set it to False since we're not using log transform in this method
        self.is_log_transformed = False
        
        # Create results dictionary
        cv_results = pd.DataFrame(grid_search.cv_results_)
        
        # Return the results with all metrics
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results,
            'model': best_model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'med_ae': med_ae,
            'med_ape': med_ape,
            'test_predictions': y_pred,
            'test_actual': y_test
        }
    

    def randomized_hyperparameter_search(self, df, param_distributions=None, n_iter=10, cv=5, scoring='neg_mean_squared_error'):
        """
        Tune model hyperparameters using RandomizedSearchCV
        
        Args:
            df (DataFrame): Training data
            param_distributions (dict): Parameter distributions to sample from
            n_iter (int): Number of parameter settings sampled
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            
        Returns:
            dict: Best parameters and cross-validation results
        """
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        import numpy as np
        
        # Clean the data
        if 'Salary' in df.columns:
            df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                
        # Filter out rows with invalid salary values
        df = df[df['Salary'] > 0]
        
        # Fill missing values in YearsExperience
        if 'YearsExperience' in df.columns:
            df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
        
        # Select only the columns we need
        if set(['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']).issubset(df.columns):
            df = df[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']]
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Default parameter distributions if none provided
        if param_distributions is None:
            param_distributions = {
                'n_estimators': [50, 100, 200, 300, 400, 500],
                'max_depth': [None, 10, 20, 30, 40, 50],
                'min_samples_split': [2, 5, 10, 15, 20],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'max_features': ['sqrt', 'log2', None]
            }
        
        # Remove random_state from param_distributions if it exists
        if 'random_state' in param_distributions:
            del param_distributions['random_state']
        
        # Create base model with fixed random_state
        rf = RandomForestRegressor(random_state=42)
        
        # Create RandomizedSearchCV object
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # Use all available cores
            verbose=1,
            random_state=42,
            return_train_score=True
        )
        
        # Fit the random search
        random_search.fit(X_scaled, y)
        
        # Get best parameters and score
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        
        # Add random_state to best_params
        best_params['random_state'] = 42
        
        # Create a model with the best parameters
        best_model = RandomForestRegressor(**best_params)
        best_model.fit(X_scaled, y)
        
        # Save the tuned model
        self.model = best_model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.is_loaded = True
        
        # Set log transformation flag to False for this method
        self.is_log_transformed = False
        
        # Create results dictionary
        cv_results = pd.DataFrame(random_search.cv_results_)
        
        # Return the results
        return {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': cv_results,
            'model': best_model
        }
    
    def train_model_with_log_transform(self, df, params, test_size=0.2):
        """
        Train a salary prediction model using log-transformed target values
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            test_size (float): Test split ratio
            
        Returns:
            dict: Model performance metrics
        """
        # Clean the data
        if 'Salary' in df.columns:
            df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
            df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                
        # Filter out rows with invalid salary values
        df = df[df['Salary'] > 0]
        
        # Fill missing values in YearsExperience
        if 'YearsExperience' in df.columns:
            df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
        
        # Select only the columns we need
        if set(['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']).issubset(df.columns):
            df = df[['YearsExperience', 'Education', 'JobRole', 'Age', 'Gender', 'Salary']]
        
        # Prepare data
        X = df.drop('Salary', axis=1)
        y = df['Salary']
        
        # Log-transform the target variable
        y_log = np.log1p(y)
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train_log, y_test_log = train_test_split(
            X, y_log, test_size=test_size, random_state=params['random_state']
        )
        
        # Also keep original y values for evaluation
        _, _, y_train, y_test = train_test_split(
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
        
        # Train model on log-transformed target
        model = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=max_features,
            random_state=params['random_state']
        )
        
        model.fit(X_train_scaled, y_train_log)
        
        # Predict and transform back to original scale
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.expm1(y_pred_log)
        
        # Evaluate model on original scale
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Save model artifacts
        self.model = model
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.is_loaded = True
        self.is_log_transformed = True  # Flag to indicate log transformation was used
        
        # Return metrics and feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape,
            'feature_importance': feature_importance,
            'log_transformed': True
        }
        
        return metrics


    def train_model_with_engineered_features(self, df, params, test_size=0.2, use_log_transform=True):
        """
        Train a Random Forest model with engineered features for better prediction accuracy
        
        Args:
            df (DataFrame): Training data
            params (dict): Model parameters
            test_size (float): Test split ratio
            use_log_transform (bool): Whether to use log transformation for target
            
        Returns:
            dict: Model performance metrics
        """
        try:
            print("Starting model.train_model_with_engineered_features")
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"Parameters: {params}")
            
            # Clean the data
            if 'Salary' in df.columns:
                print("Cleaning Salary column")
                df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
                df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                print(f"Salary column cleaned. Sample values: {df['Salary'].head()}")
                    
            # Filter out rows with invalid salary values
            print("Filtering out invalid salary values")
            df = df[df['Salary'] > 0]
            print(f"After filtering, DataFrame shape: {df.shape}")
            
            # Fill missing values in YearsExperience
            if 'YearsExperience' in df.columns:
                print("Filling missing values in YearsExperience")
                df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
            
            # Engineer features
            print("Engineering features")
            df_engineered = self.engineer_features(df)
            print(f"Engineered DataFrame shape: {df_engineered.shape}")
            print(f"Engineered columns: {df_engineered.columns.tolist()}")
            
            # Prepare data
            print("Preparing data for training")
            X = df_engineered.drop('Salary', axis=1)
            y = df_engineered['Salary']
            
            # Log-transform the target variable if requested
            if use_log_transform:
                print("Applying log transformation to target")
                y_log = np.log1p(y)
            else:
                y_log = y
            
            # Handle categorical variables
            print("Handling categorical variables")
            categorical_cols = X.select_dtypes(include=['object']).columns
            print(f"Categorical columns: {categorical_cols.tolist()}")
            label_encoders = {}
            
            for col in categorical_cols:
                print(f"Encoding column: {col}")
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Split data
            print("Splitting data")
            X_train, X_test, y_train_log, y_test_log = train_test_split(
                X, y_log, test_size=test_size, random_state=params['random_state']
            )
            print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
            
            # Also keep original y values for evaluation
            _, _, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=params['random_state']
            )
            
            # Scale features
            print("Scaling features")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fix max_features parameter
            max_features = params.get('max_features', 'sqrt')
            if max_features == 'auto':
                max_features = 'sqrt'  # 'auto' is deprecated, use 'sqrt' instead
            
            # Optimize Random Forest parameters
            print("Setting up Random Forest parameters")
            rf_params = {
                'n_estimators': params.get('n_estimators', 200),
                'max_depth': params.get('max_depth', None),
                'min_samples_split': params.get('min_samples_split', 5),
                'min_samples_leaf': params.get('min_samples_leaf', 2),
                'max_features': max_features,
                'bootstrap': True,
                'random_state': params.get('random_state', 42),
                'n_jobs': -1
            }
            
            # Remove oob_score and max_samples if they cause issues
            # rf_params['oob_score'] = True
            # rf_params['max_samples'] = 0.8
            
            print(f"Random Forest parameters: {rf_params}")
            
            # Train Random Forest model with optimized parameters
            print("Training Random Forest model")
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train_scaled, y_train_log)
            print("Model training completed")
            
            # Predict and transform back to original scale if needed
            print("Making predictions")
            y_pred_log = model.predict(X_test_scaled)
            if use_log_transform:
                y_pred = np.expm1(y_pred_log)
            else:
                y_pred = y_pred_log
            
            # Evaluate model on original scale
            print("Evaluating model")
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate additional metrics
            mae = np.mean(np.abs(y_test - y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            print(f"Metrics - RMSE: {rmse}, R2: {r2}, MAE: {mae}, MAPE: {mape}")
            
            # Save model artifacts
            print("Saving model artifacts")
            self.model = model
            self.scaler = scaler
            self.label_encoders = label_encoders
            self.is_loaded = True
            self.is_log_transformed = use_log_transform
            self.uses_engineered_features = True
            
            # Save feature names for later use in prediction
            self.feature_names = X.columns.tolist()
            
            # Get feature importance
            print("Calculating feature importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            # Calculate OOB score if available
            oob_score = None
            if hasattr(model, 'oob_score_'):
                oob_score = model.oob_score_
                print(f"OOB Score: {oob_score}")
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'oob_score': oob_score,
                'feature_importance': feature_importance,
                'log_transformed': use_log_transform,
                'engineered_features': True,
                'model_type': 'random_forest',
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print("Returning metrics")
            return metrics
            
        except Exception as e:
            print(f"Error in model.train_model_with_engineered_features: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to be caught by the caller
    # def train_model_with_engineered_features(self, df, params, test_size=0.2, use_log_transform=True):
    #     """
    #     Train a model with engineered features for better prediction accuracy
        
    #     Args:
    #         df (DataFrame): Training data
    #         params (dict): Model parameters
    #         test_size (float): Test split ratio
    #         use_log_transform (bool): Whether to use log transformation for target
            
    #     Returns:
    #         dict: Model performance metrics
    #     """
    #     # Clean the data
    #     if 'Salary' in df.columns:
    #         df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
    #         df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                
    #     # Filter out rows with invalid salary values
    #     df = df[df['Salary'] > 0]
        
    #     # Fill missing values in YearsExperience
    #     if 'YearsExperience' in df.columns:
    #         df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].median())
        
    #     # Engineer features
    #     df_engineered = self.engineer_features(df)
        
    #     # Prepare data
    #     X = df_engineered.drop('Salary', axis=1)
    #     y = df_engineered['Salary']
        
    #     # Log-transform the target variable if requested
    #     if use_log_transform:
    #         y_log = np.log1p(y)
    #     else:
    #         y_log = y
        
    #     # Handle categorical variables
    #     categorical_cols = X.select_dtypes(include=['object']).columns
    #     label_encoders = {}
        
    #     for col in categorical_cols:
    #         le = LabelEncoder()
    #         X[col] = le.fit_transform(X[col])
    #         label_encoders[col] = le
        
    #     # Split data
    #     X_train, X_test, y_train_log, y_test_log = train_test_split(
    #         X, y_log, test_size=test_size, random_state=params['random_state']
    #     )
        
    #     # Also keep original y values for evaluation
    #     _, _, y_train, y_test = train_test_split(
    #         X, y, test_size=test_size, random_state=params['random_state']
    #     )
        
    #     # Scale features
    #     scaler = StandardScaler()
    #     X_train_scaled = scaler.fit_transform(X_train)
    #     X_test_scaled = scaler.transform(X_test)
        
    #     # Fix max_features parameter
    #     max_features = params['max_features']
    #     if max_features == 'auto':
    #         max_features = 'sqrt'  # 'auto' is deprecated, use 'sqrt' instead
        
    #     # Train model
    #     model = RandomForestRegressor(
    #         n_estimators=params['n_estimators'],
    #         max_depth=params['max_depth'],
    #         min_samples_split=params['min_samples_split'],
    #         min_samples_leaf=params['min_samples_leaf'],
    #         max_features=max_features,
    #         random_state=params['random_state']
    #     )
        
    #     model.fit(X_train_scaled, y_train_log)
        
    #     # Predict and transform back to original scale if needed
    #     y_pred_log = model.predict(X_test_scaled)
    #     if use_log_transform:
    #         y_pred = np.expm1(y_pred_log)
    #     else:
    #         y_pred = y_pred_log
        
    #     # Evaluate model on original scale
    #     mse = mean_squared_error(y_test, y_pred)
    #     rmse = np.sqrt(mse)
    #     r2 = r2_score(y_test, y_pred)
        
    #     # Calculate additional metrics
    #     mae = np.mean(np.abs(y_test - y_pred))
    #     mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
    #     # Save model artifacts
    #     self.model = model
    #     self.scaler = scaler
    #     self.label_encoders = label_encoders
    #     self.is_loaded = True
    #     self.is_log_transformed = use_log_transform
    #     self.uses_engineered_features = True
        
    #     # Save feature names for later use in prediction
    #     self.feature_names = X.columns.tolist()
        
    #     # Return metrics and feature importance
    #     feature_importance = pd.DataFrame({
    #         'Feature': X.columns,
    #         'Importance': model.feature_importances_
    #     }).sort_values(by='Importance', ascending=False)
        
    #     metrics = {
    #         'mse': mse,
    #         'rmse': rmse,
    #         'r2': r2,
    #         'mae': mae,
    #         'mape': mape,
    #         'feature_importance': feature_importance,
    #         'log_transformed': use_log_transform,
    #         'engineered_features': True
    #     }
        
    #     return metrics

    # def engineer_features(self, df):
    #     """
    #     Engineer features to improve Random Forest model performance
        
    #     Args:
    #         df (DataFrame): Input data
            
    #     Returns:
    #         DataFrame: Data with engineered features
    #     """
    #     try:
    #         print("Starting engineer_features")
    #         print(f"Input DataFrame shape: {df.shape}")
    #         print(f"Input DataFrame columns: {df.columns.tolist()}")
            
    #         # Make a copy to avoid modifying the original
    #         df_new = df.copy()
            
    #         # 1. Experience transformations (polynomial features)
    #         print("Creating experience transformations")
    #         df_new['Experience_Squared'] = df_new['YearsExperience'] ** 2
    #         df_new['Experience_Cubed'] = df_new['YearsExperience'] ** 3
    #         df_new['Log_Experience'] = np.log1p(df_new['YearsExperience'])
            
    #         # Create experience buckets with finer granularity
    #         try:
    #             print("Creating experience buckets")
    #             df_new['Experience_Bucket'] = pd.cut(
    #                 df_new['YearsExperience'], 
    #                 bins=[0, 1, 3, 5, 7, 10, 15, float('inf')], 
    #                 labels=[0, 1, 2, 3, 4, 5, 6]
    #             )
    #         except Exception as e:
    #             print(f"Error creating experience buckets with pd.cut: {e}")
    #             # If there's an error with pd.cut, use a simpler approach
    #             df_new['Experience_Bucket'] = df_new['YearsExperience'].apply(
    #                 lambda x: 0 if x < 1 else (1 if x < 3 else (2 if x < 5 else 
    #                                         (3 if x < 7 else (4 if x < 10 else 
    #                                                             (5 if x < 15 else 6)))))
    #             )
            
    #         # 2. Age-related features
    #         print("Creating age-related features")
    #         df_new['Age_Squared'] = df_new['Age'] ** 2
    #         df_new['Age_to_Experience'] = df_new['Age'] / (df_new['YearsExperience'] + 1)
    #         df_new['Experience_to_Age_Ratio'] = df_new['YearsExperience'] / df_new['Age']
            
    #         # Career stage (early, mid, senior) based on age and experience
    #         df_new['Career_Stage'] = 0  # Default: early career
    #         mid_career_mask = (df_new['Age'] >= 30) & (df_new['YearsExperience'] >= 5)
    #         senior_mask = (df_new['Age'] >= 35) & (df_new['YearsExperience'] >= 10)
    #         df_new.loc[mid_career_mask, 'Career_Stage'] = 1  # Mid career
    #         df_new.loc[senior_mask, 'Career_Stage'] = 2  # Senior
            
    #         # 3. Tech stack features (if available)
    #         if 'LanguageHaveWorkedWith' in df_new.columns:
    #             print("Creating tech stack features")
    #             # Count number of languages
    #             df_new['Language_Count'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
    #                 lambda x: len(str(x).split(';')) if str(x).strip() else 0
    #             )
                
    #             # Check for specific in-demand languages
    #             in_demand_langs = ['Python', 'JavaScript', 'Java', 'C#', 'Go', 'TypeScript', 'SQL']
    #             for lang in in_demand_langs:
    #                 df_new[f'Knows_{lang}'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
    #                     lambda x: 1 if lang in str(x) else 0
    #                 )
                
    #             # Create language category features
    #             df_new['Knows_WebTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
    #                 lambda x: 1 if any(lang in str(x) for lang in ['JavaScript', 'HTML', 'CSS', 'TypeScript']) else 0
    #             )
    #             df_new['Knows_DataTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
    #                 lambda x: 1 if any(lang in str(x) for lang in ['Python', 'R', 'SQL', 'Scala']) else 0
    #             )
    #             df_new['Knows_MobileTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
    #                 lambda x: 1 if any(lang in str(x) for lang in ['Swift', 'Kotlin', 'Objective-C']) else 0
    #             )
            
    #         # 4. Education level as numerical with more weight to higher degrees
    #         print("Creating education level features")
    #         education_mapping = {
    #             'High School': 0,
    #             'Bachelor': 1,
    #             'Master': 3,  # Increased weight
    #             'PhD': 6      # Increased weight
    #         }
            
    #         # Create a copy of Education for mapping
    #         if 'Education' in df_new.columns:
    #             df_new['Education_Level'] = df_new['Education'].map(education_mapping).fillna(1)
            
    #         # 5. Job role complexity with more nuanced weights
    #         print("Creating job role complexity features")
    #         role_complexity = {
    #             'Front-end Developer': 1.0,
    #             'Back-end Developer': 1.5,
    #             'Full-stack Developer': 2.0,
    #             'Mobile Developer': 1.8,
    #             'DevOps': 2.2,
    #             'Data Scientist': 2.5,
    #             'Embedded Engineer': 2.0,
    #             'Game Developer': 2.0
    #         }
            
    #         if 'JobRole' in df_new.columns:
    #             df_new['Role_Complexity'] = df_new['JobRole'].map(role_complexity).fillna(1.5)
            
    #         # 6. Interaction features (more sophisticated)
    #         print("Creating interaction features")
    #         # Education × Experience with polynomial terms
    #         df_new['Education_x_Experience'] = df_new['Education_Level'] * df_new['YearsExperience']
    #         df_new['Education_x_Experience_Squared'] = df_new['Education_Level'] * df_new['Experience_Squared']
            
    #         # Role complexity × Experience with polynomial terms
    #         df_new['Role_x_Experience'] = df_new['Role_Complexity'] * df_new['YearsExperience']
    #         df_new['Role_x_Experience_Squared'] = df_new['Role_Complexity'] * df_new['Experience_Squared']
            
    #         # Education × Role complexity (higher education in complex roles often pays more)
    #         df_new['Education_x_Role'] = df_new['Education_Level'] * df_new['Role_Complexity']
            
    #         # 7. Company size as numerical (if available)
    #         if 'OrgSize' in df_new.columns:
    #             print("Creating company size features")
    #             # Map company size to numerical values with exponential scaling
    #             size_mapping = {
    #                 'Less than 20 employees': 10,
    #                 '2-9 employees': 5,
    #                 '10 to 19 employees': 15,
    #                 '20 to 99 employees': 60,
    #                 '100 to 499 employees': 300,
    #                 '500 to 999 employees': 750,
    #                 '1,000 to 4,999 employees': 3000,
    #                 '5,000 to 9,999 employees': 7500,
    #                 '10,000 or more employees': 15000
    #             }
    #             df_new['OrgSize_Numeric'] = df_new['OrgSize'].map(size_mapping).fillna(60)
                
    #             # Log transform company size (often has non-linear relationship with salary)
    #             df_new['Log_OrgSize'] = np.log1p(df_new['OrgSize_Numeric'])
                
    #             # Interaction between company size and experience
    #             df_new['OrgSize_x_Experience'] = df_new['OrgSize_Numeric'] * df_new['YearsExperience'] / 1000  # Scale down
            
    #         # 8. Gender as numerical
    #         if 'Gender' in df_new.columns:
    #             print("Creating gender features")
    #             gender_mapping = {
    #                 'Male': 0,
    #                 'Female': 1,
    #                 'Other': 2
    #             }
    #             df_new['Gender_Numeric'] = df_new['Gender'].map(gender_mapping).fillna(0)
            
    #         # 9. Domain expertise approximation (years in role)
    #         print("Creating domain expertise features")
    #         # Assume someone with more experience in a complex role has more domain expertise
    #         df_new['Domain_Expertise'] = df_new['YearsExperience'] * df_new['Role_Complexity'] / 2
            
    #         # 10. Senior level indicator (boolean)
    #         df_new['Is_Senior'] = (df_new['YearsExperience'] >= 8).astype(int)
            
    #         print(f"Engineered DataFrame shape: {df_new.shape}")
    #         print(f"Engineered DataFrame columns: {df_new.columns.tolist()}")
            
    #         return df_new
            
    #     except Exception as e:
    #         print(f"Error in engineer_features: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise  # Re-raise the exception to be caught by the caller

    def engineer_features(self, df):
        """
        Engineer features to improve Random Forest model performance
        
        Args:
            df (DataFrame): Input data
            
        Returns:
            DataFrame: Data with engineered features
        """
        try:
            print("Starting engineer_features")
            print(f"Input DataFrame shape: {df.shape}")
            print(f"Input DataFrame columns: {df.columns.tolist()}")
            
            # Make a copy to avoid modifying the original
            df_new = df.copy()
            
            # 1. Experience transformations (polynomial features)
            print("Creating experience transformations")
            df_new['Experience_Squared'] = df_new['YearsExperience'] ** 2
            df_new['Experience_Cubed'] = df_new['YearsExperience'] ** 3
            df_new['Log_Experience'] = np.log1p(df_new['YearsExperience'])
            
            # Create experience buckets with finer granularity
            try:
                print("Creating experience buckets")
                df_new['Experience_Bucket'] = pd.cut(
                    df_new['YearsExperience'], 
                    bins=[0, 1, 3, 5, 7, 10, 15, float('inf')], 
                    labels=[0, 1, 2, 3, 4, 5, 6]
                )
            except Exception as e:
                print(f"Error creating experience buckets with pd.cut: {e}")
                # If there's an error with pd.cut, use a simpler approach
                df_new['Experience_Bucket'] = df_new['YearsExperience'].apply(
                    lambda x: 0 if x < 1 else (1 if x < 3 else (2 if x < 5 else 
                                            (3 if x < 7 else (4 if x < 10 else 
                                                                (5 if x < 15 else 6)))))
                )
            
            # 2. Age-related features
            print("Creating age-related features")
            df_new['Age_Squared'] = df_new['Age'] ** 2
            df_new['Age_to_Experience'] = df_new['Age'] / (df_new['YearsExperience'] + 1)
            df_new['Experience_to_Age_Ratio'] = df_new['YearsExperience'] / df_new['Age']
            
            # Career stage (early, mid, senior) based on age and experience
            df_new['Career_Stage'] = 0  # Default: early career
            mid_career_mask = (df_new['Age'] >= 30) & (df_new['YearsExperience'] >= 5)
            senior_mask = (df_new['Age'] >= 35) & (df_new['YearsExperience'] >= 10)
            df_new.loc[mid_career_mask, 'Career_Stage'] = 1  # Mid career
            df_new.loc[senior_mask, 'Career_Stage'] = 2  # Senior
            
            # 3. Tech stack features (if available)
            if 'LanguageHaveWorkedWith' in df_new.columns:
                print("Creating tech stack features")
                # Count number of languages
                df_new['Language_Count'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
                    lambda x: len(str(x).split(';')) if str(x).strip() else 0
                )
                
                # Check for specific in-demand languages
                in_demand_langs = ['Python', 'JavaScript', 'Java', 'C#', 'Go', 'TypeScript', 'SQL']
                for lang in in_demand_langs:
                    df_new[f'Knows_{lang}'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
                        lambda x: 1 if lang in str(x) else 0
                    )
                
                # Create language category features
                df_new['Knows_WebTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
                    lambda x: 1 if any(lang in str(x) for lang in ['JavaScript', 'HTML', 'CSS', 'TypeScript']) else 0
                )
                df_new['Knows_DataTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
                    lambda x: 1 if any(lang in str(x) for lang in ['Python', 'R', 'SQL', 'Scala']) else 0
                )
                df_new['Knows_MobileTech'] = df_new['LanguageHaveWorkedWith'].fillna('').apply(
                    lambda x: 1 if any(lang in str(x) for lang in ['Swift', 'Kotlin', 'Objective-C']) else 0
                )
            
            # 4. Education level as numerical with more weight to higher degrees
            print("Creating education level features")
            education_mapping = {
                'High School': 0,
                'Bachelor': 1,
                'Master': 3,  # Increased weight
                'PhD': 6      # Increased weight
            }
            
            # Create a copy of Education for mapping
            if 'Education' in df_new.columns:
                df_new['Education_Level'] = df_new['Education'].map(education_mapping).fillna(1)
            
            # 5. Job role complexity with more nuanced weights
            print("Creating job role complexity features")
            role_complexity = {
                'Front-end Developer': 1.0,
                'Back-end Developer': 1.5,
                'Full-stack Developer': 2.0,
                'Mobile Developer': 1.8,
                'DevOps': 2.2,
                'Data Scientist': 2.5,
                'Embedded Engineer': 2.0,
                'Game Developer': 2.0
            }
            
            if 'JobRole' in df_new.columns:
                df_new['Role_Complexity'] = df_new['JobRole'].map(role_complexity).fillna(1.5)
            
            # 6. Interaction features (more sophisticated)
            print("Creating interaction features")
            # Education × Experience with polynomial terms
            df_new['Education_x_Experience'] = df_new['Education_Level'] * df_new['YearsExperience']
            df_new['Education_x_Experience_Squared'] = df_new['Education_Level'] * df_new['Experience_Squared']
            
            # Role complexity × Experience with polynomial terms
            df_new['Role_x_Experience'] = df_new['Role_Complexity'] * df_new['YearsExperience']
            df_new['Role_x_Experience_Squared'] = df_new['Role_Complexity'] * df_new['Experience_Squared']
            
            # Education × Role complexity (higher education in complex roles often pays more)
            df_new['Education_x_Role'] = df_new['Education_Level'] * df_new['Role_Complexity']
            
            # 7. Company size as numerical (if available)
            if 'OrgSize' in df_new.columns:
                print("Creating company size features")
                # Map company size to numerical values with exponential scaling
                size_mapping = {
                    'Less than 20 employees': 10,
                    '2-9 employees': 5,
                    '10 to 19 employees': 15,
                    '20 to 99 employees': 60,
                    '100 to 499 employees': 300,
                    '500 to 999 employees': 750,
                    '1,000 to 4,999 employees': 3000,
                    '5,000 to 9,999 employees': 7500,
                    '10,000 or more employees': 15000
                }
                df_new['OrgSize_Numeric'] = df_new['OrgSize'].map(size_mapping).fillna(60)
                
                # Log transform company size (often has non-linear relationship with salary)
                df_new['Log_OrgSize'] = np.log1p(df_new['OrgSize_Numeric'])
                
                # Interaction between company size and experience
                df_new['OrgSize_x_Experience'] = df_new['OrgSize_Numeric'] * df_new['YearsExperience'] / 1000  # Scale down
            
            # 8. Gender as numerical
            if 'Gender' in df_new.columns:
                print("Creating gender features")
                gender_mapping = {
                    'Male': 0,
                    'Female': 1,
                    'Other': 2
                }
                df_new['Gender_Numeric'] = df_new['Gender'].map(gender_mapping).fillna(0)
            
            # 9. Domain expertise approximation (years in role)
            print("Creating domain expertise features")
            # Assume someone with more experience in a complex role has more domain expertise
            df_new['Domain_Expertise'] = df_new['YearsExperience'] * df_new['Role_Complexity'] / 2
            
            # 10. Senior level indicator (boolean)
            df_new['Is_Senior'] = (df_new['YearsExperience'] >= 8).astype(int)
            
            print(f"Engineered DataFrame shape: {df_new.shape}")
            print(f"Engineered DataFrame columns: {df_new.columns.tolist()}")
            
            return df_new
            
        except Exception as e:
            print(f"Error in engineer_features: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception to be caught by the caller
    
    def predict_salary_with_engineered_features(self, features):
        """
        Predict salary using a model trained with engineered features
        
        Args:
            features (dict): Dictionary with feature values
            
        Returns:
            float: Predicted salary in VND
        """
        # Create DataFrame with single row for feature engineering
        # Map the input feature keys to the column names expected by engineer_features
        input_df = pd.DataFrame([{
            'YearsExperience': features['experience'],
            'Education': features['education'],
            'JobRole': features['job_role'],
            'Age': features['age'],
            'Gender': features['gender']
        }])
        
        # Engineer features
        try:
            input_engineered = self.engineer_features(input_df)
            
            # Encode categorical features
            for col in input_engineered.select_dtypes(include=['object']).columns:
                if col in self.label_encoders:
                    input_engineered[col] = self.label_encoders[col].transform(input_engineered[col])
            
            # Ensure all required columns are present
            if hasattr(self, 'feature_names'):
                missing_cols = set(self.feature_names) - set(input_engineered.columns)
                for col in missing_cols:
                    input_engineered[col] = 0  # Default value for missing columns
                
                # Select only the columns used during training
                input_engineered = input_engineered[self.feature_names]
            
            # Scale features
            input_scaled = self.scaler.transform(input_engineered)
            
            # Make prediction
            prediction = self.model.predict(input_scaled)[0]
            
            # Transform back if log transformed
            if hasattr(self, 'is_log_transformed') and self.is_log_transformed:
                prediction = np.expm1(prediction)
                
            return prediction
        except Exception as e:
            print(f"Error in predict_salary_with_engineered_features: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to regular prediction or mock
            if self.is_loaded:
                # Regular prediction without engineered features
                input_dict = {
                    'YearsExperience': features['experience'],
                    'Education': self.label_encoders['Education'].transform([features['education']])[0],
                    'JobRole': self.label_encoders['JobRole'].transform([features['job_role']])[0],
                    'Age': features['age'],
                    'Gender': self.label_encoders['Gender'].transform([features['gender']])[0],
                }
                input_df = pd.DataFrame([input_dict])
                input_scaled = self.scaler.transform(input_df)
                
                # Make prediction
                prediction = self.model.predict(input_scaled)[0]
                
                # If model was trained with log-transformed targets, transform back
                if hasattr(self, 'is_log_transformed') and self.is_log_transformed:
                    prediction = np.expm1(prediction)
                    
                return prediction
            else:
                # Mock prediction for demo
                return self._mock_predict(features)
        
    def tune_hyperparameters_with_engineered_features(self, df, param_grid=None, cv=5, scoring='neg_mean_squared_error', use_log_transform=True):
        """
        Tune model hyperparameters using GridSearchCV with engineered features
        
        Args:
            df (DataFrame): Training data
            param_grid (dict): Grid of parameters to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric for evaluation
            use_log_transform (bool): Whether to use log transformation for target
            
        Returns:
            dict: Best parameters and cross-validation results
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np
        
        try:
            print("Starting tune_hyperparameters_with_engineered_features")
            
            # Clean the data
            if 'Salary' in df.columns:
                df['Salary'] = df['Salary'].astype(str).str.replace(',', '').str.replace('"', '')
                df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
                    
            # Filter out rows with invalid salary values
            df = df[df['Salary'] > 0]
            
            # Engineer features
            df_engineered = self.engineer_features(df)
            
            # Prepare data
            X = df_engineered.drop('Salary', axis=1)
            y = df_engineered['Salary']
            
            # Log-transform the target variable if requested
            if use_log_transform:
                y = np.log1p(y)
            
            # Handle categorical variables
            categorical_cols = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Split data for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Default parameter grid if none provided
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            
            # Create base model
            rf = RandomForestRegressor(random_state=42)
            
            # Create GridSearchCV object
            grid_search = GridSearchCV(
                estimator=rf,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,  # Use all available cores
                verbose=1,
                return_train_score=True
            )
            
            # Fit the grid search
            grid_search.fit(X_train_scaled, y_train)
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Create a model with the best parameters
            best_model = RandomForestRegressor(**best_params, random_state=42)
            best_model.fit(X_train_scaled, y_train)
            
            # Make predictions on test set
            y_pred = best_model.predict(X_test_scaled)

            # Calculate metrics in log space (if using log transform)
            if use_log_transform:
                log_mse = mean_squared_error(y_test, y_pred)
                log_rmse = np.sqrt(log_mse)
                log_r2 = r2_score(y_test, y_pred)
                log_mae = mean_absolute_error(y_test, y_pred)

                # Transform back to original scale
                y_test_original = np.expm1(y_test)
                y_pred_original = np.expm1(y_pred)
            else:
                y_test_original = y_test
                y_pred_original = y_pred

            # Calculate metrics in original space
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred_original)
            mae = mean_absolute_error(y_test_original, y_pred_original)
            mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

            # Add median-based metrics
            med_ae = np.median(np.abs(y_test_original - y_pred_original))
            med_ape = np.median(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
            
            # Save the tuned model
            self.model = best_model
            self.scaler = scaler
            self.label_encoders = label_encoders
            self.is_loaded = True
            self.is_log_transformed = use_log_transform
            self.uses_engineered_features = True
            self.feature_names = X.columns.tolist()
            
            # Create results dictionary
            cv_results = pd.DataFrame(grid_search.cv_results_)
            
            # Return the results with all metrics
            return {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': cv_results,
                'model': best_model,
                'feature_names': X.columns.tolist(),
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': mae,
                'mape': mape,
                'med_ae': med_ae,
                'med_ape': med_ape,
                # Include log-space metrics if using log transform
                'log_mse': log_mse if use_log_transform else None,
                'log_rmse': log_rmse if use_log_transform else None,
                'log_r2': log_r2 if use_log_transform else None,
                'log_mae': log_mae if use_log_transform else None,
                'test_predictions': y_pred_original,
                'test_actual': y_test_original
            }
        
        except Exception as e:
            import traceback
            print(f"Error in tune_hyperparameters_with_engineered_features: {e}")
            print(traceback.format_exc())
            raise

    def save_tuning_results(self, tuning_results, version=None):
        """
        Save hyperparameter tuning results along with the model
        
        Args:
            tuning_results (dict): Results from hyperparameter tuning
            version (int, optional): Version number to save as
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("Starting save_tuning_results")
            
            # If version is not provided, use the current version
            if version is None:
                version = self.version
            
            # If no version is set yet, return False
            if version is None:
                print("No version specified and no current version set")
                return False
            
            # Get the path to the version directory
            version_path = os.path.join(self.model_dir, f'v{version}')
            
            # Ensure the directory exists
            if not os.path.exists(version_path):
                print(f"Version directory {version_path} does not exist")
                return False
            
            # Convert DataFrame to dict for serialization if needed
            if 'cv_results' in tuning_results and isinstance(tuning_results['cv_results'], pd.DataFrame):
                print("Converting cv_results DataFrame to dict")
                tuning_results['cv_results'] = tuning_results['cv_results'].to_dict()
            
            # Remove model object as it's already saved
            if 'model' in tuning_results:
                print("Removing model object from tuning results")
                del tuning_results['model']
            
            # Save tuning results
            print(f"Saving tuning results to {version_path}")
            joblib.dump(tuning_results, os.path.join(version_path, 'tuning_results.pkl'))
            
            # Save feature names if available
            if hasattr(self, 'feature_names'):
                print("Updating metadata with feature names")
                try:
                    metadata_path = os.path.join(version_path, 'metadata.pkl')
                    if os.path.exists(metadata_path):
                        metadata = joblib.load(metadata_path)
                        metadata['engineered_feature_names'] = self.feature_names
                        
                        # Add metrics to metadata for display in the UI
                        if 'best_score' in tuning_results:
                            if 'metrics' not in metadata:
                                metadata['metrics'] = {}
                            
                            # Calculate and add common metrics
                            metadata['metrics']['best_score'] = float(tuning_results['best_score'])
                            
                            # If we have test predictions, calculate more detailed metrics
                            if 'test_predictions' in tuning_results and 'test_actual' in tuning_results:
                                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                                import numpy as np
                                
                                y_test = tuning_results['test_actual']
                                y_pred = tuning_results['test_predictions']
                                
                                # Calculate metrics
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)
                                mae = mean_absolute_error(y_test, y_pred)
                                
                                # Calculate MAPE
                                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                                
                                # Add metrics to metadata
                                metadata['metrics']['mse'] = float(mse)
                                metadata['metrics']['rmse'] = float(rmse)
                                metadata['metrics']['r2'] = float(r2)
                                metadata['metrics']['mae'] = float(mae)
                                metadata['metrics']['mape'] = float(mape)
                        
                        # Save updated metadata
                        joblib.dump(metadata, metadata_path)
                except Exception as e:
                    print(f"Error updating metadata: {e}")
            
            print("Tuning results saved successfully")
            return True
        
        except Exception as e:
            print(f"Error saving tuning results: {e}")
            import traceback
            traceback.print_exc()
            return False
    

    # @staticmethod
    # def train_model_with_engineered_features(df, params, save_as_version=None, use_log_transform=True):
    #     """
    #     Train a model with engineered features for better prediction accuracy
        
    #     Args:
    #         df (DataFrame): Training data
    #         params (dict): Model parameters
    #         save_as_version (int, optional): Save as specific version number
    #         use_log_transform (bool): Whether to use log transformation for target
            
    #     Returns:
    #         dict: Model performance metrics and new version number
    #     """
    #     # Remove Location column if it exists
    #     if 'Location' in df.columns:
    #         df = df.drop('Location', axis=1)
            
    #     model = SalaryPredictionModel()
    #     metrics = model.train_model_with_engineered_features(df, params, test_size=params.get('test_size', 0.2), use_log_transform=use_log_transform)
        
    #     # Extract metrics for saving (exclude feature_importance as it's not serializable)
    #     save_metrics = {k: v for k, v in metrics.items() if k != 'feature_importance'}
        
    #     # Save the model with specific version if provided
    #     success = model.save_model(version=save_as_version, metrics=save_metrics)
    #     if success:
    #         version = model.version
    #         metrics['version'] = version
    #         metrics['success'] = True
    #         metrics['message'] = f"Model v{version} with engineered features {'updated' if save_as_version else 'created'} successfully"
    #     else:
    #         metrics['success'] = False
    #         metrics['message'] = "Failed to save model"
            
    #     return metrics

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
        
        # Remove Location column if it exists
        if 'Location' in df.columns:
            df = df.drop('Location', axis=1)
            
        model = SalaryPredictionModel()
        
        # Run hyperparameter tuning with engineered features
        if method == "grid":
            tuning_results = model.tune_hyperparameters_with_engineered_features(
                df, 
                param_grid=param_grid, 
                cv=cv, 
                scoring=scoring,
                use_log_transform=use_log_transform
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
                'cv_results_fig': fig if 'cv_results_fig' in tuning_results else None,
                'engineered_features': True
            }
        else:
            return {
                'success': False,
                'message': "Failed to save model"
            }