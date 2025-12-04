# RF-VN Employee Salary Predictor

A comprehensive machine learning system for predicting software engineer salaries in Vietnam using Random Forest regression models. Built with Streamlit, featuring role-based authentication, model versioning, and advanced feature engineering.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [User Roles & Workflows](#user-roles--workflows)
- [Model Features & Training](#model-features--training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [License](#license)

## Overview

The RF-VN Employee Salary Predictor is an intelligent system designed to predict software engineer salaries in Vietnam based on multiple factors including education, experience, job role, and demographic information. The system provides both single predictions and batch processing capabilities, with a sophisticated admin interface for model management and monitoring.

### Key Capabilities

- **Accurate Salary Predictions**: Uses trained Random Forest models with engineered features
- **Model Versioning**: Manage multiple model versions with version switching and comparison
- **Batch Processing**: Predict salaries for multiple employees via CSV upload
- **Role-Based Access**: Separate interfaces for administrators and regular users
- **Advanced Analytics**: Feature importance analysis, prediction metrics, and data visualization
- **Hyperparameter Tuning**: Built-in grid search and randomized search for model optimization

## Features

### User-Facing Features

- **User Authentication**: Secure login system with role-based access control
- **Individual Salary Prediction**: Input employee details to get salary predictions
- **Batch Salary Prediction**: Upload CSV files containing employee data for bulk predictions
- **Admin Dashboard**: Comprehensive admin interface for system management
- **Data Visualization**: Interactive charts showing feature importance and model performance
- **Model Management**: View, select, and manage different model versions

### Technical Features

- **Feature Engineering**: Advanced feature creation including polynomial terms, interaction features, and domain expertise metrics
- **Log Transformation**: Optional log transformation of target variables for better model fitting
- **Hyperparameter Optimization**: GridSearchCV and RandomizedSearchCV for hyperparameter tuning
- **Multiple Training Strategies**: Standard training, training with log transformation, and engineered features training
- **Model Persistence**: Model versioning system with metadata tracking and serialization using joblib

## System Architecture

The application follows a **Model-View-Controller (MVC)** architecture with clear separation of concerns:

```
predictor_system/
├── models/                    # Machine Learning Models
│   ├── salary_model.py       # Core SalaryPredictionModel class
│   ├── user.py              # User data model
│   └── *.ipynb              # Data science notebooks for exploration
├── views/                    # Streamlit User Interfaces
│   ├── auth_view.py        # Login & authentication UI
│   ├── admin_view.py       # Admin dashboard & management UI
│   └── user_view.py        # User salary prediction UI
├── controllers/             # Business Logic Layer
│   ├── auth_controller.py  # Authentication logic
│   ├── admin_controller.py # Admin operations
│   └── user_controller.py  # User operations & predictions
├── config/                  # Configuration
│   ├── settings.py         # App configuration & constants
│   └── __init__.py
├── data/                    # Data Management
│   ├── sample_data.py      # Sample data utilities
│   ├── preprocessing/      # Data preprocessing notebooks
│   └── processed_data/     # Processed datasets
├── model_versions/          # Model Storage
│   ├── v1/, v14/, v15/...  # Versioned models with artifacts
│   └── Contains: salary_model.pkl, scaler.pkl, label_encoders.pkl, metadata.pkl
├── utils/                  # Utility Functions
│   ├── data_utils.py      # Data manipulation utilities
│   └── visualization.py   # Plotting & visualization functions
├── app.py                 # Main Streamlit application
├── app_settings.json      # Runtime settings & active model version
└── employee_salary_data.csv # Sample data
```

### Architecture Layers

#### 1. **Data Layer** (`data/` & `models/`)
- Handles data loading, preprocessing, and cleaning
- Manages raw and processed datasets
- Provides sample data for demonstrations

#### 2. **Model Layer** (`models/salary_model.py`)
- `SalaryPredictionModel`: Core class for salary prediction
- Supports multiple training approaches:
  - Standard Random Forest training
  - Training with log-transformed targets
  - Training with engineered features
- Implements feature engineering with polynomial and interaction terms
- Manages model versioning and persistence
- Supports batch and single predictions

#### 3. **Business Logic Layer** (`controllers/`)
- **AuthController**: Handles user authentication and role verification
- **UserController**: Manages user operations including predictions
- **AdminController**: Handles administrative tasks and model management

#### 4. **Presentation Layer** (`views/` & `app.py`)
- **auth_view.py**: Role selection and login interface
- **admin_view.py**: Dashboard, user management, data visualization, model management
- **user_view.py**: Salary prediction interface (single & batch)
- **app.py**: Main application router and session state manager

#### 5. **Configuration Layer** (`config/`)
- User credentials and role definitions
- Model parameters and default settings
- Feature definitions for predictions
- Currency conversion settings (INR to VND)

### Data Flow

```
User Input
    ↓
Streamlit View (UI)
    ↓
Controller (Business Logic)
    ↓
Model (SalaryPredictionModel)
    ↓
Prediction/Result
    ↓
View (Display Results)
```

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Package Manager**: pip (or conda)
- **OS**: macOS, Linux, or Windows

### Step 1: Clone Repository

```bash
git clone https://github.com/LyMinhTrungitdsiu19023/RF-VN-Employee-Salary-Predictor.git
cd RF-VN-Employee-Salary-Predictor
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using Python venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n salary-predictor python=3.9
conda activate salary-predictor
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models
- `joblib` - Model serialization
- `matplotlib` - Data visualization

### Step 4: Verify Installation

```bash
python -c "import streamlit; import pandas; import sklearn; print('All dependencies installed successfully')"
```

## Running the Application

### Basic Start

```bash
streamlit run predictor_system/app.py
```

The application will start on `http://localhost:8501` and automatically open in your default browser.

### Advanced Options

```bash
# Run with specific configuration
streamlit run predictor_system/app.py --logger.level=debug

# Run on specific port
streamlit run predictor_system/app.py --server.port 8502

# Run in headless mode (no browser auto-open)
streamlit run predictor_system/app.py --server.headless true
```

## User Roles & Workflows

### 1. Admin Users

**Login Credentials:**
- Username: `admin1`, `admin2`, or `admin3`
- Passwords: `admin123`, `admin456`, or `admin789` (respectively)

**Admin Dashboard Features:**

#### Dashboard Home
- System overview and statistics
- Quick access to all admin functions

#### User Management
- View all users in the system
- Add new users with specific roles
- Edit user information
- Delete users
- Manage user permissions

#### Data Visualization
- Feature importance charts (showing which factors most influence salary predictions)
- Model performance metrics
- Salary distribution analysis
- Experience vs. Salary correlation plots

#### Model Management
- View all available model versions (v1, v14, v15, v16, etc.)
- Switch between different model versions
- View model parameters and configuration
- Monitor model performance metrics
- Delete outdated model versions
- Train new models with custom parameters

### 2. Regular Users

**Sample Login Credentials:**
- Username: `student1`, Password: `stud123`
- Username: `engineer1`, Password: `eng123`
- Username: `recruiter1`, Password: `rec123`

**User Features:**

#### Single Salary Prediction
1. Enter employee details:
   - Years of Experience (0-50 years)
   - Education Level (High School, Bachelor, Master, PhD)
   - Job Role (8 specialized roles)
   - Age (18-70 years)
   - Gender (Male, Female, Other)
2. Click "Predict Salary"
3. View predicted salary in Vietnamese Dong (₫)

#### Batch Prediction
1. Prepare CSV file with columns:
   - `YearsExperience` (numeric)
   - `Education` (string: High School, Bachelor, Master, PhD)
   - `JobRole` (string: from predefined list)
   - `Age` (numeric)
   - `Gender` (string: Male, Female, Other)

2. Upload CSV file via the interface
3. System processes all records
4. Download results as CSV with predictions

**Supported Job Roles:**
- Data Scientist
- Back-end Developer
- Front-end Developer
- Mobile Developer
- Embedded Engineer
- DevOps
- Full-stack Developer
- Game Developer

## Model Features & Training

### Input Features for Predictions

| Feature | Type | Range/Values | Impact |
|---------|------|-------------|--------|
| Years Experience | Numeric | 0-50 | High - Strong salary correlation |
| Education | Categorical | High School, Bachelor, Master, PhD | High - Education level directly impacts salary |
| Job Role | Categorical | 8 specialized roles | High - Different roles have different pay scales |
| Age | Numeric | 18-70 | Medium - Proxy for experience and career stage |
| Gender | Categorical | Male, Female, Other | Low to Medium - Demographic factor |

### Engineered Features

The advanced model training includes feature engineering that creates:

1. **Experience Features:**
   - Squared and cubic terms (non-linear relationships)
   - Log-transformed experience
   - Experience buckets/categories

2. **Age Features:**
   - Age squared (polynomial term)
   - Age-to-experience ratio
   - Career stage classification (Early, Mid, Senior)

3. **Role Features:**
   - Role complexity scoring
   - Role-experience interactions
   - Role-education interactions

4. **Education Features:**
   - Education level encoding with weighted values
   - Education-experience interaction terms
   - Education-role interaction terms

5. **Domain Features:**
   - Domain expertise scores
   - Senior level indicators
   - Technology stack knowledge (when available)

### Model Training Methods

#### 1. Standard Training
- Basic Random Forest without transformations
- Use when data has normal distribution
- Faster training and prediction

#### 2. Log Transformation Training
- Applies log transformation to target variable (salary)
- Better for skewed salary distributions
- Helps reduce impact of extreme values
- Requires inverse transformation for final predictions

#### 3. Engineered Features Training
- Uses all advanced feature engineering techniques
- Longer training time but often better accuracy
- Captures non-linear relationships and interactions
- Most suitable for complex salary patterns

### Hyperparameter Optimization

The system supports two optimization strategies:

#### Grid Search CV
- Exhaustive search over specified parameter grid
- Tests all combinations
- More thorough but computationally intensive

#### Randomized Search CV
- Random sampling of parameter space
- Faster than grid search
- Good for large parameter spaces

**Tunable Parameters:**
- `n_estimators`: Number of trees (50-500)
- `max_depth`: Maximum tree depth (None, 10-50)
- `min_samples_split`: Minimum samples to split (2-20)
- `min_samples_leaf`: Minimum samples in leaf (1-8)
- `max_features`: Feature selection (sqrt, log2, None)

### Model Persistence

Each model version is stored in `model_versions/vX/` containing:

- `salary_model.pkl` - Trained Random Forest model
- `scaler.pkl` - Feature scaler (StandardScaler)
- `label_encoders.pkl` - Categorical feature encoders
- `metadata.pkl` - Version metadata including:
  - Creation timestamp
  - Feature names
  - Model type and parameters
  - Performance metrics
  - Feature importance scores
  - Log transformation flag
  - Engineered features flag

## Project Structure

```
RF-VN-Employee-Salary-Predictor/
├── predictor_system/
│   ├── __init__.py
│   ├── app.py                          # Main Streamlit app
│   ├── app_settings.json               # Runtime configuration
│   ├── employee_salary_data.csv        # Sample dataset
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                 # App settings & credentials
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── salary_model.py             # Core prediction model
│   │   ├── user.py                     # User data model
│   │   ├── new_data.csv                # Test data
│   │   └── data_scientist_cleaning_and_evaluation.ipynb
│   │
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── auth_controller.py          # Authentication logic
│   │   ├── admin_controller.py         # Admin operations
│   │   └── user_controller.py          # User operations
│   │
│   ├── views/
│   │   ├── __init__.py
│   │   ├── auth_view.py                # Login UI
│   │   ├── admin_view.py               # Admin dashboard UI
│   │   └── user_view.py                # User interface UI
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_utils.py               # Data utilities
│   │   └── visualization.py            # Plotting utilities
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sample_data.py              # Sample data provider
│   │   ├── preprocessing/              # Data preprocessing
│   │   │   ├── preprocessing.ipynb
│   │   │   └── process.ipynb
│   │   └── processed_data/             # Processed datasets
│   │       ├── combined_salary_data_2017_2025.csv
│   │       ├── combined_salary_data_2021_2025.csv
│   │       ├── processed_salary_data_standardized.csv
│   │       ├── refined_salary_data.csv
│   │       └── backup_*.csv
│   │
│   └── model_versions/                 # Versioned models
│       ├── v1/
│       ├── v14/
│       ├── v15/
│       ├── v16/
│       └── vX/
│           ├── salary_model.pkl
│           ├── scaler.pkl
│           ├── label_encoders.pkl
│           ├── metadata.pkl
│           └── tuning_results.pkl
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── .gitignore                         # Git ignore rules
```

## Configuration

### App Settings (`predictor_system/config/settings.py`)

#### User Credentials
```python
USERS = {
    "admin1": {"password": "admin123", "role": "Admin"},
    "student1": {"password": "stud123", "role": "User", "user_type": "Student"},
    "engineer1": {"password": "eng123", "role": "User", "user_type": "Software Engineer"},
    # ... more users
}
```

#### Feature Options
```python
EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']
JOB_ROLES = ['Data Scientist', 'Back-end Developer', 'Front-end Developer', ...]
GENDERS = ['Male', 'Female', 'Other']
```

#### Currency Settings
```python
CURRENCY_SYMBOL_VND = '₫'    # Display symbol
```

### Runtime Settings (`predictor_system/app_settings.json`)

```json
{
  "active_model_version": 16
}
```

Stores the currently active model version selected by admins.

## Usage Examples

### Example 1: Single Salary Prediction

1. Start the application
2. Select "User" role
3. Login with credentials (e.g., `engineer1` / `eng123`)
4. Enter employee information:
   - Years Experience: 5
   - Education: Master
   - Job Role: Back-end Developer
   - Age: 28
   - Gender: Male
5. Click "Predict Salary"
6. View predicted salary (e.g., 45,000,000 ₫)

### Example 2: Batch Prediction

1. Prepare CSV file (`employees.csv`):
   ```
   YearsExperience,Education,JobRole,Age,Gender
   5,Master,Back-end Developer,28,Male
   8,Bachelor,Data Scientist,32,Female
   3,Bachelor,Front-end Developer,25,Female
   ```

2. Login as regular user
3. Navigate to "Batch Prediction"
4. Upload CSV file
5. Click "Predict"
6. Download results with predictions

### Example 3: Admin Model Management

1. Login as admin (`admin1` / `admin123`)
2. Navigate to "Model Management"
3. View available versions and their metrics
4. Select v16 as active model
5. Review feature importance and performance metrics

## Development & Model Training

### Using Jupyter Notebooks

Data science work is documented in notebooks:
- `predictor_system/models/data_scientist_cleaning_and_evaluation.ipynb` - Data exploration and model evaluation
- `predictor_system/data/preprocessing/preprocessing.ipynb` - Data preprocessing pipeline
- `predictor_system/data/preprocessing/process.ipynb` - Advanced data processing

### Training New Models

To train and save new model versions:

1. Prepare training data in `predictor_system/data/processed_data/`
2. Use admin interface "Model Management" → "Train New Model"
3. Or programmatically:

```python
from predictor_system.models.salary_model import SalaryPredictionModel
import pandas as pd

# Load data
df = pd.read_csv('path/to/data.csv')

# Initialize model
model = SalaryPredictionModel()

# Train with engineered features
metrics = model.train_model_with_engineered_features(
    df,
    params={
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }
)

# Save model
model.save_model(version=17)
```

## Performance Metrics

The system tracks multiple evaluation metrics:

- **MSE (Mean Squared Error)**: Average squared differences
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **MAE (Mean Absolute Error)**: Average absolute differences
- **R² Score**: Proportion of variance explained (0-1 scale)
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error

Lower MSE/RMSE/MAE values indicate better performance. Higher R² values (closer to 1.0) indicate better fit.

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Ensure you're in the project directory
cd RF-VN-Employee-Salary-Predictor
python -c "import predictor_system"
```

**2. Port Already in Use**
```bash
# Use different port
streamlit run predictor_system/app.py --server.port 8502
```

**3. Missing CSV Columns in Batch Prediction**
- Ensure CSV has exact column names: `YearsExperience`, `Education`, `JobRole`, `Age`, `Gender`
- Check for typos and proper capitalization

**4. Model Loading Issues**
- Verify model files exist in `model_versions/vX/`
- Check `app_settings.json` for valid version number

## License

Copyright (c) 2025 University of Technology - Ho Chi Minh City Vietnam National University

Intelligence System Course

## Contributors

- **Ly Minh Trung** - Lead Developer & Data Scientist
  - Email: itdsiu19023@student.hcmut.edu.vn
  - GitHub: [@LyMinhTrungitdsiu19023](https://github.com/LyMinhTrungitdsiu19023)

## Acknowledgments

- Vietnam National University, Ho Chi Minh City
- University of Technology Faculty
- Machine Learning and Intelligence Systems Course

---

For questions, issues, or contributions, please open an issue on GitHub or contact the developers.