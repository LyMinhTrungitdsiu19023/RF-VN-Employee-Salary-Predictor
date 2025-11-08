import streamlit as st

# Predefined user accounts - in a real app, this would be in a database
USERS = {
    # Admin accounts
    "admin1": {"password": "admin123", "role": "Admin"},
    "admin2": {"password": "admin456", "role": "Admin"},
    "admin3": {"password": "admin789", "role": "Admin"},
    
    # User accounts - Updated user types
    "student1": {"password": "stud123", "role": "User", "user_type": "Student"},
    "engineer1": {"password": "eng123", "role": "User", "user_type": "Software Engineer"},
    "recruiter1": {"password": "rec123", "role": "User", "user_type": "Recruiter"},
}

# Feature options for salary prediction
EDUCATION_LEVELS = ['High School', 'Bachelor', 'Master', 'PhD']
JOB_ROLES = ['Data Scientist', 'Software Engineer', 'Manager', 'Analyst', 'HR']
GENDERS = ['Male', 'Female', 'Other']

# Updated user types
USER_TYPES = ["Student", "Software Engineer", "Recruiter"]

# Currency conversion settings (module-level so other modules can import them)
# Approximate conversion rate: 1 INR -> 280 VND (update if you need a different rate)
RATE_INR_TO_VND = 280
CURRENCY_SYMBOL_VND = '₫'

def set_page_config():
    """Set the page configuration for the Streamlit app"""
    st.set_page_config(
        page_title="Employee System",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
        <style>
        .stButton button {
            height: 3rem;
            font-size: 1.2rem;
        }
        .admin-card {
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    