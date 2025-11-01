import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define possible values
education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
job_roles = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
locations = ['New York', 'San Francisco', 'Austin', 'Remote', 'India']
genders = ['Male', 'Female', 'Other']

# Generate synthetic data
n_samples = 500
data = []
for _ in range(n_samples):
    experience = np.random.randint(0, 21)  # 0-20 years
    education = random.choice(education_levels)
    role = random.choice(job_roles)
    location = random.choice(locations)
    age = np.random.randint(22, 61)  # 22-60 years
    gender = random.choice(genders)
    # Base salary in INR for 0 experience
    base_salary_inr = 150000
    increment_per_year = 200000  # INR per year of experience
    salary_inr = base_salary_inr + (experience * increment_per_year)
    # Education bonus
    if education == 'Bachelor':
        salary_inr += 400000
    elif education == 'Master':
        salary_inr += 800000
    elif education == 'PhD':
        salary_inr += 1600000
    # Role bonus
    if role == 'Manager':
        salary_inr += 1200000
    elif role == 'Data Scientist':
        salary_inr += 1000000
    elif role == 'Software Engineer':
        salary_inr += 800000
    elif role == 'Analyst':
        salary_inr += 500000
    # Location adjustment
    if location == 'San Francisco':
        salary_inr += 800000
    elif location == 'New York':
        salary_inr += 600000
    elif location == 'Austin':
        salary_inr += 300000
    elif location == 'India':
        salary_inr -= 1000000  # Lower cost of living
    # Add limited noise
    salary_inr += int(np.random.normal(0, 100000))
    salary_inr = max(salary_inr, 1000000)  # Ensure minimum salary
    data.append([
        experience, education, role, location, age, gender, salary_inr
    ])

# Create DataFrame
columns = ['YearsExperience', 'Education', 'JobRole', 'Location', 'Age', 'Gender', 'Salary']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
csv_path = 'employee_salary_data.csv'
df.to_csv(csv_path, index=False)

print(f"Dataset generated and saved as {csv_path}. You can now download this file.")

# --- Data Exploration & Preprocessing ---
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
print("\n--- Data Exploration ---")
df = pd.read_csv(csv_path)
print(df.head())
print(df.describe())
print("\nMissing values per column:\n", df.isnull().sum())

# Encode categorical variables
label_encoders = {}
for col in ['Education', 'JobRole', 'Location', 'Gender']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
y = df['Salary']
X = df.drop('Salary', axis=1)

# Feature scaling (optional, good for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2 Score: {r2:.2f}")

# Save the model and scaler
joblib.dump(model, 'salary_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
print("Trained model, scaler, and encoders saved for later use.")
