import numpy as np
import pandas as pd
import os
from config.settings import EDUCATION_LEVELS, JOB_ROLES, GENDERS

def create_sample_data(n_samples=500, save_to_csv=False):
    """
    Create sample employee salary data for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        save_to_csv (bool): Whether to save the generated data to CSV
        
    Returns:
        DataFrame: Generated sample data
    """
    np.random.seed(42)
    
    # Generate features
    years_exp = np.random.uniform(0, 20, n_samples)
    age = np.random.randint(22, 60, n_samples)
    
    # Categorical features
    education = np.random.choice(EDUCATION_LEVELS, n_samples)
    job_role = np.random.choice(JOB_ROLES, n_samples)
    gender = np.random.choice(GENDERS, n_samples)
    
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
    
    age_factor = (age - 22) * 1000
    
    # Add some random noise
    noise = np.random.normal(0, 10000, n_samples)
    
    # Calculate final salary
    salary = base_salary + exp_factor + edu_factor + role_factor + age_factor + noise
    salary = np.round(salary).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'YearsExperience': years_exp,
        'Education': education,
        'JobRole': job_role,
        'Age': age,
        'Gender': gender,
        'Salary': salary
    })
    
    # Save to CSV if requested
    if save_to_csv:
        df.to_csv('employee_salary_data.csv', index=False)
    
    return df