# Employee Salary Prediction App

This project predicts employee salaries using a machine learning model trained on synthetic data. The app is built with Streamlit and allows both manual and batch predictions.

## Features
- Manual salary prediction via form inputs
- Batch prediction via CSV upload
- Uses a Random Forest Regressor trained on synthetic employee data

## Project Structure
- `EMP SAL.py`: Script to generate data and train the model
- `employee_salary_data.csv`: Synthetic dataset
- `salary_model.pkl`, `scaler.pkl`, `label_encoders.pkl`: Model artifacts
- `emp_salary_app.py`: Streamlit app

## Setup Instructions

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Generate data and train model**
   If you want to regenerate the dataset and retrain the model, run:
   ```bash
   python "EMP SAL.py"
   ```
   This will overwrite the CSV and model artifacts.

3. **Run the Streamlit app**
   ```bash
   streamlit run emp_salary_app.py
   ```

4. **Using the App**
   - Fill in the form for manual prediction, or
   - Upload a CSV with columns: `YearsExperience, Education, JobRole, Location, Age, Gender` for batch prediction.

## Notes
- Ensure `salary_model.pkl`, `scaler.pkl`, and `label_encoders.pkl` are present in the same directory as `emp_salary_app.py`.
- The app uses synthetic data and is for demonstration purposes only. 