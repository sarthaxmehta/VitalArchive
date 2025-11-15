import streamlit as st
import datetime
import pandas as pd
import os


st.set_page_config(
    page_title="Register",
    page_icon="📝",
)

st.title("Create a New Account")


with st.form("registration_form"):
    st.header("Registration Details")

   
    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth", 
                        min_value=datetime.date(1920, 1, 1), 
                        max_value=datetime.date.today())
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

    st.markdown("---")

   
    new_password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Your Password", type="password")

    
    submitted = st.form_submit_button("Register")

    if submitted:
        
        if not all([name, dob, gender, new_password, confirm_password]):
            st.warning("Please fill out all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match. Please try again.")
        else:
            
            try:
                
                output_dir = "user_data"
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, "all_users_details.csv")

                new_username = 1000 
                existing_df = None
                
                
                if os.path.exists(file_path):
                    try:
                        existing_df = pd.read_csv(file_path)
                        
                        if not existing_df.empty and 'Username' in existing_df.columns:
                            
                            existing_df['Username'] = pd.to_numeric(existing_df['Username'])
                            new_username = existing_df['Username'].max() + 1
                    except pd.errors.EmptyDataError:
                        
                        existing_df = pd.DataFrame() 
                
                
                new_user_data = {
                    'Name': [name],
                    'Username': [new_username],
                    
                    'Password': [new_password], 
                    'Date of Birth': [str(dob)],
                    'Gender': [gender]
                }
                new_df = pd.DataFrame(new_user_data)

                
                if existing_df is not None and not existing_df.empty:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df

                
                combined_df.to_csv(file_path, index=False)
                
                st.success(f"Registration successful! Your new username is: **{new_username}**")
                st.info(f"All user data is now stored in: {file_path}")
                st.info("Navigate to the 'Health Report Login' page from the sidebar to sign in.")

            except Exception as e:
                st.error(f"An error occurred while saving your data: {e}")



st.markdown("---")
st.page_link("report.py", label="Already have an account? Login", icon="👋")

