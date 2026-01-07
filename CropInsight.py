import streamlit as st

st.set_page_config(page_title="Crop Insight", layout="wide")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login form
if not st.session_state.logged_in:
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()  # <-- FORCE Streamlit to refresh
        else:
            st.error("Invalid credentials")

# After login
if st.session_state.logged_in:
    st.title("✅ Welcome to Crop Insight Dashboard")
    st.write("Use the sidebar to navigate to different pages")
    # Optionally, show dashboard overview here
