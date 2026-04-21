import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

def run_page():
    ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def load_segments():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            all_names = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl')
                all_names.extend(df.iloc[::2, 0].dropna().unique().tolist())
            return sorted([n for n in all_names if str(n).lower() not in ['city', 'category', 'segment', 'status']])
        except: return ["Hyderabad", "Delhi"]

    all_segments = load_segments()

    # --- UI STYLING (Preserved) ---
    st.markdown("<style>.stButton>button { width: 100%; background-color: #059669; color: white; font-weight: bold; }</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.title("🛡️ Code X Engine")
        kw = st.text_input("Keywords", key="x_kw")
        desc = st.text_area("Product Description")
        goal = st.text_area("Intention")
        
        # LINKED MULTI-SELECT
        st.session_state.selected_segments = st.multiselect(
            "Target Segments", options=all_segments, default=st.session_state.selected_segments
        )
        circle_on = st.checkbox("CIRCLE Subscriber", value=False)

    st.title("🚀 Code X: Performance Synthesis")
    
    # DNA UPLOADERS (Preserved)
    st.markdown("### 📂 DNA Streams")
    col1, col2 = st.columns(2)
    s1_files = col1.file_uploader("Performance DNA (CSV)", accept_multiple_files=True)
    s2_file = col2.file_uploader("Style DNA (CSV)")

    if st.button("🚀 MASTER GENERATE"):
        if not st.session_state.selected_segments:
            st.error("Select segments first!")
        else:
            try:
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                # Passing the full list of segments to the AI
                prompt = f"AUDIENCE: {st.session_state.selected_segments}. PROD: {desc}. GOAL: {goal}. KW: {kw}. CIRCLE: {circle_on}."
                res = model.generate_content(prompt)
                st.markdown(res.text)
            except Exception as e: st.error(f"Error: {e}")
