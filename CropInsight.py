import streamlit as st

st.set_page_config(page_title="Crop Recommendation System", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

st.title("🔐 Login")

if not st.session_state.logged_in:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    st.success("You are logged in!")
    st.write("Use the sidebar to navigate the system.")

