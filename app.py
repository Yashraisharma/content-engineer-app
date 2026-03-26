import streamlit as st
import pandas as pd
from google import genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)
st.set_page_config(page_title="Sure-Shot Attribution Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #28a745; color: white; font-weight: bold; border: none; }
    section[data-testid="stSidebar"] { width: 450px !important; }
    .logic-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #d63384; font-size: 1.1em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & TARGETING ---
with st.sidebar:
    st.title("🛡️ Segment Engineer Pro")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="Refill, Safety, Insulin")
    prod_description = st.text_area("Product Details", height=120)
    intention = st.text_area("Primary Goal", height=120)

    st.divider()
    
    st.header("🔍 Advanced Targeting (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed Users")
    seg_reason = st.text_input("2. Reason for Segment", placeholder="e.g. 30-Day Inactive")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus")

    st.divider()
    
    st.header("⚙️ Applied Ranking Logic")
    with st.expander("🔬 View Formula", expanded=True):
        st.markdown('<div class="logic-box"><b>Sure-Shot Score:</b><br><span class="formula">(CTR*0.6) + (LogScale*0.2) + (Rel*0.2)</span></div>', unsafe_allow_html=True)
        st.caption("CTR measures efficiency; LogScale ensures statistical confidence; Rel matches your Who Query.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Deep-Column Campaign Attribution & Engineering")

uploaded_files = st.file_uploader("Upload Campaign CSVs (CleverTap/MoEngage Header Format)", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # HEADER MAPPING
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        sent_col = col_map.get('total sent(users)', None)
        clicks_col = col_map.get('total clicked(users)', None)

        if all([id_col, msg_col, sent_col, clicks_col]):
            # A. DATA CLEANING
            df['Sent_N'] = pd.to_numeric(df[sent_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # B. CALCULATE COMPONENTS
            # 1. CTR (Efficiency)
            df['CTR_Calc'] = (df['Clicks_N'] / df['Sent_N']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # 2. Log Scale (Confidence) - Normalized 0 to 1
            max_sent = df['Sent_N'].max() if df['Sent_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Sent_N']) / np.log1p(max_sent)
            
            # 3. Relevance (Segment Alignment)
            target_str = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower()
            if who_col and target_str.strip():
                df['Rel_Match'] = df[who_col].apply(lambda x: 1.0 if any(k in str(x).lower() for k in target_str.split()) else 0.0)
            else:
                df['Rel_Match'] = 0.5 # Neutral

            # C. FINAL SURE-SHOT RANKING
            df['Attribution_Score'] = (df['CTR_Calc'] * 0.6) + (df['Log_Scale'] * 0.2) + (df['Rel_Match'] * 0.2)

            # --- DISPLAY ---
            st.subheader("🔍 Step 1: Raw Campaign Audit (All Columns)")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.subheader("🏆 Step 2: Performance Ranking (The DNA Blueprints)")
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            st.table(winners[[id_col, who_col, msg_col, 'CTR_Calc', 'Attribution_Score']])

            # --- STEP 3: CONTENT ENGINEERING WITH 429 PROTECTION ---
            if st.button("🚀 Step 3: Engineer 10 Attribution-Mapped Variations"):
                if not ACTIVE_KEY:
                    st.error("API Key missing! Add GEMINI_API_KEY to Streamlit Secrets.")
                else:
                    try:
                        client = genai.Client(api_key=ACTIVE_KEY)
                        
                        # Build context string for AI
                        context = ""
                        for _, row in winners.iterrows():
                            context += f"ID: {row[id_col]} | Who: {row.get(who_col)} | CTR: {row['CTR_Calc']:.4%} | Msg: {row[msg_col]}\n"

                        prompt = f"""
                        You are a Behavioral Growth Specialist. 
                        NEW TARGET: {seg_type} | {seg_reason} | {sub_seg} | {spec_prod}
                        PRODUCT: {prod_description} | GOAL: {intention} | KEYWORDS: {keywords_input}
                        
                        HISTORICAL WINNERS (RANKED BY SURE-SHOT LOGIC):
                        {context}
                        
                        TASK:
                        1. Factual Audit: Briefly explain why these IDs succeeded for their Who Queries.
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Every row must show: Reference Campaign ID, New Content, Source Segment, Hit %, Reasoning.
                        """
                        
                        with st.spinner("Analyzing and Engineering... (API Request Sent)"):
                            response = client.models.generate_content(model="gemini-3-flash", contents=prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            st.error("⚠️ **Quota Reached (20/day):** You've hit the Gemini Free Tier limit. Results cannot be generated until tomorrow.")
                        else:
                            st.error(f"System Error: {e}")
        else:
            st.error("Header Error: Required columns (Campaign ID, Message, Total Sent, Total Clicked) not found.")

    except Exception as e:
        st.error(f"Critical Error: {e}")
else:
    st.info("👋 Upload CSVs to perform deep-column attribution and segment targeting.")
