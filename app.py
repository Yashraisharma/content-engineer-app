import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Growth Content Engineer Tool", layout="wide")

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ Strategic Command")
st.sidebar.subheader("Apollo 247 Growth Ops")

page_choice = st.sidebar.radio(
    "Select Module", 
    ["Code X (Generator)", "Segment Analysis & Predictor"]
)

# --- ROUTING LOGIC ---
if page_choice == "Code X (Generator)":
    try:
        import code_x
        code_x.run_page()
    except ImportError:
        st.error("Error: 'code_x.py' not found in your GitHub repository.")

elif page_choice == "Segment Analysis & Predictor":
    try:
        import predictor
        predictor.run_page()
    except ImportError:
        st.error("Error: 'predictor.py' not found in your GitHub repository.")
