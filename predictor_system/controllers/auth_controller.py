import streamlit as st
from models.user import UserModel

class AuthController:
    """Controller for authentication-related operations"""
    
    @staticmethod
    def login(username, password, selected_role=None, user_type=None):
        """
        Authenticate a user and update session state
        
        Args:
            username (str): Username
            password (str): Password
            selected_role (str, optional): Role selected by user
            user_type (str, optional): User type for new users
            
        Returns:
            dict: Login result with success status and message
        """
        # First, validate credentials without checking role
        login_result = UserModel.validate_login(username, password, None, user_type)
        
        if login_result["success"]:
            # Get the actual role from the validation result
            actual_role = login_result.get("role")
            
            # Store user info in session state
            st.session_state.username = username
            st.session_state.role = actual_role
            st.session_state.user_type = login_result.get("user_type")
            
            # If user selected the wrong role, inform them but still log them in
            if selected_role and actual_role != selected_role:
                login_result["message"] = f"Note: Your account is a {actual_role} account, not a {selected_role} account. Logging you in as {actual_role}."
            
            # Set the appropriate redirect page
            if actual_role == "User":
                st.session_state.page = 'salary_prediction'
            else:
                st.session_state.page = 'dashboard'
        
        return login_result
    
    @staticmethod
    def logout():
        """Log out the current user by resetting session state"""
        # Save the active model version before clearing
        active_model = st.session_state.get('active_model_version')
        
        # Clear session state
        st.session_state.page = 'role_selection'
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.user_type = None
        
        # Restore the active model version
        if active_model:
            st.session_state.active_model_version = active_model