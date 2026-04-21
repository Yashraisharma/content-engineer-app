import streamlit as st

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Apollo 247 Growth Command", layout="wide")

# --- INITIALIZE GLOBAL MEMORY (Session State) ---
if "selected_segments" not in st.session_state:
    st.session_state.selected_segments = []

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ Strategic Command")
st.sidebar.caption("Senior Growth Content Engineer")

page_choice = st.sidebar.radio(
    "Select Module", 
    ["Code X (Generator)", "Segment Analysis & Predictor"]
)

# --- PAGE ROUTING ---
if page_choice == "Code X (Generator)":
    try:
        import code_x
        code_x.run_page()
    except Exception as e:
        st.error(f"Error loading Code X: {e}")

elif page_choice == "Segment Analysis & Predictor":
    try:
        import predictor
        predictor.run_page()
    except Exception as e:
        st.error(f"Error loading Predictor: {e}")
