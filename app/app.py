# app/app.py
import streamlit as st

st.set_page_config(page_title="Seal Identifier 🦭", layout="centered")

pages = [
    st.Page("predictions.py", title="Predictions", icon="🔍", default=True),
    # st.Page("queue.py", title="Training Queue", icon="📋"),   # future tabs go here
]
st.navigation(pages, position="top").run()
