import streamlit as st
import pandas as pd
import os


st.set_page_config(
    page_title="Health Report Login",
    page_icon="👋",
    layout="wide"  # Use wide layout for a two-column design
)


st.markdown("""
    <style>
    /* This targets the specific column to center its contents */
    div[data-testid="column"]:nth-of-type(1) div[data-testid="stVerticalBlock"] {
        gap: 0rem;
    }
    </style>
""", unsafe_allow_html=True)


USER_DATA_PATH = os.path.join("user_data", "all_users_details.csv")


def load_user_data_and_check():
    """
    Loads user data from CSV into session_state once and returns a 
    function to check credentials against the in-memory data.
    """
    
    if 'users_loaded' not in st.session_state:
        st.session_state.users = {}
        if os.path.exists(USER_DATA_PATH):
            try:
                df = pd.read_csv(USER_DATA_PATH)
                if not df.empty and 'Username' in df.columns:
                    # Convert usernames to strings for consistent key access
                    df['Username'] = df['Username'].astype(str)
                    st.session_state.users = df.set_index('Username').T.to_dict()
            except (pd.errors.EmptyDataError, Exception) as e:
                st.error(f"Error loading user data: {e}")
        st.session_state.users_loaded = True
    
    
    def check_credentials(username, password):
        """Checks if the username and password are valid."""
        if username not in st.session_state.users:
            return False, "User ID not found. Please register or check your User ID."
        
       
        if str(st.session_state.users[username].get('Password')) == str(password):
            user_full_name = st.session_state.users[username].get('Name', 'User')
            return True, user_full_name
        else:
            return False, "Invalid password. Please try again."

    return check_credentials


if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['name'] = None


credential_checker = load_user_data_and_check()


if not st.session_state['logged_in']:
    col1, col2 = st.columns([1, 1], gap="large")

    
    with col1:
        st.title("Health Report Analysis")
        st.header("Login to Your Account")

        username_input = st.text_input("Username", placeholder="Enter your username")
        password_input = st.text_input("Password", type="password", placeholder="Enter your password")

        login_button = st.button("Login", use_container_width=True)

        if login_button:
            is_valid, result = credential_checker(username_input, password_input)
            if is_valid:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username_input
                st.session_state['name'] = result  
                st.rerun()
            else:
                st.error(result)  
        
        st.markdown("---")
        st.page_link("pages/_register.py", label="New user? Register here", icon="📝")

    
    with col2:
        image_url = "https://images.unsplash.com/photo-1576091160550-2173dba999ef?q=80&w=2070&auto=format&fit=crop"
        st.image(image_url, caption="Your Health, Your Data", use_container_width=True)


if st.session_state['logged_in']:
    st.sidebar.success("Select a page above.")
    st.header(f"Welcome, {st.session_state['name']}!")
    st.success("You are successfully logged in.")
    st.page_link("pages/_evaluate_Report.py", label="Go to Evaluate Report", icon="🩺")

