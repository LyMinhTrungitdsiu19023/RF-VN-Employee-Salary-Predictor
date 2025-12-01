# Employee System - Salary Predictor

## Overview

The Employee System is a web application built with Streamlit that provides salary prediction functionality. The system has different user roles (Admin and User) and allows for both individual salary predictions and batch predictions via CSV file uploads.

## Features

- **User Authentication**: Secure login system with role-based access control
- **Admin Dashboard**: User management capabilities for administrators
- **Salary Prediction**: Individual and batch salary predictions based on various features
- **Responsive UI**: Clean, modern interface with responsive design

## System Architecture

The application follows an MVC (Model-View-Controller) architecture:

- **Models**: Handle data and prediction logic
  - `SalaryPredictionModel`: Manages salary predictions based on features

- **Views**: Streamlit-based user interfaces
  - `admin_view.py`: Admin interface for user management
  - Other view modules for different functionalities

- **Controllers**: Business logic between models and views
  - `admin_controller.py`: Handles admin operations like user management
  - `user_controller.py`: Manages user operations like salary predictions

## Prediction Features

The salary prediction model uses the following features:
- Years of Experience
- Education Level
- Job Role
- Location
- Age
- Gender

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- Pandas

### Installation

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the application:


## Usage

### Admin Users

Administrators can:
- View all users in the system
- Add new users with different roles
- Manage user permissions

### Regular Users

Regular users can:
- Make individual salary predictions
- Upload CSV files for batch predictions
- View prediction results

## Batch Prediction

For batch predictions, upload a CSV file with the following columns:
- YearsExperience
- Education
- JobRole
- Location
- Age
- Gender

## License

Copyright (c) 2025 University of Technology - Ho Chi Minh City Vietnam National University

Intelligence System Course

## Contributors

Ly Minh Trung