import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

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

    # --- RESTORED CSS & STYLING ---
    st.markdown("""
        <style>
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #059669; color: white; font-weight: bold; }
        .stream-header { background-color: #0f172a; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.title("🛡️ Code X Engine")
        kw = st.text_input("Keywords", key="x_kw")
        desc = st.text_area("Product Description", height=100)
        goal = st.text_area("Intention", height=80)
        
        st.divider()
        st.session_state.selected_segments = st.multiselect(
            "Target Segments (Linked to Excel)", options=all_segments, default=st.session_state.selected_segments
        )
        circle_on = st.checkbox("CIRCLE Subscriber", value=False)

    st.title("🚀 Code X: Performance Synthesis")
    
    st.markdown('<div class="stream-header">📂 STREAM 1: Performance ROI DNA</div>', unsafe_allow_html=True)
    s1_files = st.file_uploader("Upload Performance CSVs", accept_multiple_files=True, key="s1")
    
    st.markdown('<div class="stream-header">📂 STREAM 2: Structural Style DNA</div>', unsafe_allow_html=True)
    s2_file = st.file_uploader("Upload Style CSV", key="s2")

    if st.button("🚀 MASTER GENERATE: SYNTHESIZE PERFORMANCE & STYLE"):
        if not st.session_state.selected_segments:
            st.error("Please select a target segment first.")
        else:
            try:
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = f"AUDIENCE: {st.session_state.selected_segments}. PROD: {desc}. GOAL: {goal}. KW: {kw}. CIRCLE: {circle_on}."
                with st.spinner("Analyzing DNA and generating..."):
                    res = model.generate_content(prompt)
                    st.markdown("### 🏆 Master Engineered Content Strategy")
                    st.markdown(res.text)
            except Exception as e: st.error(f"Error: {e}")
