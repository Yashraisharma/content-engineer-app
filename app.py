import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Sure-Shot Segment Engineer", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #28a745; color: white; font-weight: bold; border: none; }
    .logic-box { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: SEGMENT TARGETING & LOGIC ---
with st.sidebar:
    st.title("🎯 Segment Control Center")
    
    st.header("📍 Advanced Targeting")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed Users", key="v5_type")
    seg_reason = st.text_input("2. Reason for Segment", placeholder="e.g. 30-Day Inactive", key="v5_reason")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin", key="v5_sub")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus", key="v5_base")

    st.divider()
    
    st.header("💡 Ranking Logic")
    with st.expander("🔬 Segment-First Formula", expanded=True):
        st.markdown('''
        <div class="logic-box">
        <b>Final Score:</b><br>
        <span class="formula">(CTR * 0.5) + (SegmentMatch * 0.4) + (LogScale * 0.1)</span>
        </div>
        ''', unsafe_allow_html=True)
        st.caption("We now prioritize Segment Relevance (40%) to ensure results match your Optional Targeting boxes.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Segment-Aware Content Engineering")

col1, col2 = st.columns(2)

with col1:
    perf_files = st.file_uploader("📂 1. Upload Performance CSVs (CTR Data)", type="csv", accept_multiple_files=True, key="perf_v5")

with col2:
    format_file = st.file_uploader("📂 2. Upload Format Template (Example Style)", type="csv", key="format_v5")

if perf_files:
    try:
        # Load and Clean Performance Data
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # Load Format Template if exists
        format_context = ""
        if format_file:
            f_df = pd.read_csv(format_file)
            format_context = f_df.to_string(index=False)
            st.success("✅ Format Template Loaded: AI will replicate this style.")

        # Column Mapping
        col_map = {c.lower(): c for c in df.columns}
        id_col, msg_col = col_map.get('campaign id'), col_map.get('message')
        who_col = col_map.get('who query')
        view_col, clicks_col = col_map.get('total viewed(users)'), col_map.get('total clicked(users)')

        if all([id_col, msg_col, view_col, clicks_col]):
            # A. DATA CALCULATIONS
            df['Viewed_N'] = pd.to_numeric(df[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # B. CTR % (Clicks / Viewed)
            df['CTR_Dec'] = (df['Clicks_N'] / df['Viewed_N']).replace([np.inf, -np.inf], 0).fillna(0)
            df['CTR_Perc'] = (df['CTR_Dec'] * 100).round(2)
            
            # C. SEGMENT RELEVANCE LOGIC (40% Weight)
            target_terms = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower().split()
            if who_col and any(target_terms):
                df['Seg_Match'] = df[who_col].apply(lambda x: 1.0 if any(k in str(x).lower() for k in target_terms if k) else 0.0)
            else:
                df['Seg_Match'] = 0.5

            # D. LOG SCALE (10% Weight)
            max_v = df['Viewed_N'].max() if df['Viewed_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Viewed_N']) / np.log1p(max_v)

            # E. FINAL SCORE
            df['Final_Score'] = (df['CTR_Dec'] * 0.5) + (df['Seg_Match'] * 0.4) + (df['Log_Scale'] * 0.1)

            # --- DISPLAY ---
            st.subheader("🏆 Segment-Matched Winners")
            winners = df.sort_values(by='Final_Score', ascending=False).head(10).copy()
            winners['CTR %'] = winners['CTR_Perc'].astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'CTR %', 'Final_Score']])

            if st.button("🚀 Step 3: Engineer Segment-Specific Variations"):
                if not ACTIVE_KEY:
                    st.error("API Key Missing!")
                else:
                    try:
                        genai.configure(api_key=ACTIVE_KEY)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        perf_dna = ""
                        for _, row in winners.iterrows():
                            perf_dna += f"ID: {row[id_col]} | CTR: {row['CTR %']} | Seg: {row.get(who_col)} | Msg: {row[msg_col]}\n"

                        prompt = f"""
                        You are a Segment-First Growth Engineer. 
                        
                        TARGET SEGMENT: {seg_type} | {seg_reason} | {sub_seg} | {spec_prod}
                        
                        HISTORICAL DNA (Ranked by Segment Relevance + CTR):
                        {perf_dna}
                        
                        TEMPLATE FORMAT TO REPLICATE:
                        {format_context}
                        
                        TASK:
                        1. Factual Audit: Briefly explain how the winners successfully appealed to the '{seg_type}' segment.
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Use Emojis to match the 'Template Format' style.
                        4. Table columns: Variation #, Content, Ref ID, Segment Reason, Logic Hook.
                        """
                        
                        with st.spinner("Analyzing Segment DNA & Replicating Format..."):
                            response = model.generate_content(prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as api_err:
                        st.error(f"API Limit or Error: {api_err}")
        else:
            st.error("Missing columns in Performance CSV. Ensure ID, Message, Viewed, and Clicked exist.")

    except Exception as e:
        st.error(f"System Error: {e}")
