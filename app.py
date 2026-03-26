import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="High-CTR Engineer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; border: none; }
    section[data-testid="stSidebar"] { width: 450px !important; }
    .logic-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #007bff; font-size: 1.1em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & OPTIONAL TARGETING ---
with st.sidebar:
    st.title("🛡️ High-CTR Engineer")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="e.g. Refill, Insulin, Safety", key="kw_v3")
    prod_description = st.text_area("Product Details", height=150, key="prod_v3")
    intention = st.text_area("Primary Goal", height=150, key="goal_v3")

    st.divider()
    
    st.header("🔍 Advanced Targeting (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed Users", key="type_v3")
    seg_reason = st.text_input("2. Reason for Segment", placeholder="e.g. 30-Day Inactive", key="reason_v3")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin", key="sub_v3")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus", key="prod_base_v3")

    st.divider()
    
    st.header("⚙️ Ranking Logic")
    with st.expander("🔬 CTR Logic", expanded=True):
        st.markdown('<div class="logic-box"><b>Sure-Shot Formula:</b><br><span class="formula">CTR = (Clicks / Viewed) * 100</span><br><br><b>Weighting:</b><br>80% Engagement DNA<br>20% Volume Confidence</div>', unsafe_allow_html=True)
        st.caption("Prioritizes messages that convert visual impressions into clicks.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 High-Performance Campaign Attribution")

uploaded_files = st.file_uploader("Upload Campaign CSVs", type="csv", accept_multiple_files=True, key="uploader_v3")

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # UPDATED COLUMN MAPPING
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        view_col = col_map.get('total viewed(users)', None)
        clicks_col = col_map.get('total clicked(users)', None)

        if id_col and msg_col and view_col and clicks_col:
            # A. DATA CLEANING
            df['Viewed_N'] = pd.to_numeric(df[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # B. CALCULATE ACTUAL CTR PERCENTAGE (Clicks/Viewed)
            df['CTR_Decimal'] = (df['Clicks_N'] / df['Viewed_N']).replace([np.inf, -np.inf], 0).fillna(0)
            df['CTR_Percentage'] = (df['CTR_Decimal'] * 100).round(2)
            
            # C. VOLUME CONFIDENCE (Logarithmic scale)
            max_views = df['Viewed_N'].max() if df['Viewed_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Viewed_N']) / np.log1p(max_views)
            
            # D. FINAL ATTRIBUTION SCORE
            df['Attribution_Score'] = (df['CTR_Decimal'] * 0.8) + (df['Log_Scale'] * 0.2)

            # --- DISPLAY ---
            st.subheader("🔍 Step 1: Raw Campaign Audit")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.subheader("🏆 Step 2: High-CTR Benchmarks (Ranked by Clicks/Viewed)")
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            
            # Formatted column for display
            winners['Final CTR %'] = winners['CTR_Percentage'].astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'Final CTR %', 'Attribution_Score']])

            if st.button("🚀 Step 3: Engineer Content from High-CTR DNA", key="engineer_v3"):
                if not ACTIVE_KEY:
                    st.error("API Key missing! Add GEMINI_API_KEY to your Secrets.")
                else:
                    try:
                        genai.configure(api_key=ACTIVE_KEY)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        context_data = ""
                        for _, row in winners.iterrows():
                            context_data += f"ID: {row[id_col]} | CTR: {row['Final CTR %']} | Msg: {row[msg_col]}\n"

                        prompt = f"""
                        Analyze these High-CTR Winners for the segment: {seg_type} {sub_seg}.
                        TARGET GOAL: {intention}
                        WINNERS DATA: {context_data}
                        
                        TASK:
                        1. Factual Audit: Why did these IDs achieve high CTRs (Clicks/Viewed)?
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Table columns: Usage Rank, New Content, Reference ID, Source Segment, Hit %, Reasoning.
                        """
                        
                        with st.spinner("Analyzing high-performance DNA..."):
                            response = model.generate_content(prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as api_err:
                        if "429" in str(api_err):
                            st.error("⚠️ Quota Reached (20/day). Please try again tomorrow.")
                        else:
                            st.error(f"API Error: {api_err}")
        else:
            st.error("CSV Headers missing: Need 'Campaign ID', 'Message', 'Total Viewed(users)', and 'Total Clicked(users)'.")

    except Exception as e:
        st.error(f"Critical System Error: {e}")
else:
    st.info("👋 Upload your campaign CSVs to start the attribution engine.")    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin", key="sub_sidebar_v2")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus", key="prod_base_sidebar_v2")

    st.divider()
    
    st.header("⚙️ Ranking Logic")
    with st.expander("🔬 CTR Logic", expanded=True):
        st.markdown('<div class="logic-box"><b>Sure-Shot Formula:</b><br><span class="formula">CTR = (Clicks / Viewed) * 100</span><br><br><b>Weighting:</b><br>80% Engagement DNA<br>20% Volume Confidence</div>', unsafe_allow_html=True)
        st.caption("Focuses on messages that convert visual impressions into clicks.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 High-Performance Campaign Attribution")

uploaded_files = st.file_uploader("Upload Campaign CSVs", type="csv", accept_multiple_files=True, key="uploader_v2")

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # UPDATED COLUMN MAPPING
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        view_col = col_map.get('total viewed(users)', None)
        clicks_col = col_map.get('total clicked(users)', None)

        if id_col and msg_col and view_col and clicks_col:
            # A. DATA CLEANING
            df['Viewed_N'] = pd.to_numeric(df[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # B. CALCULATE ACTUAL CTR PERCENTAGE (Clicks/Viewed)
            df['CTR_Decimal'] = (df['Clicks_N'] / df['Viewed_N']).replace([np.inf, -np.inf], 0).fillna(0)
            df['CTR_Percentage'] = (df['CTR_Decimal'] * 100).round(2)
            
            # C. VOLUME CONFIDENCE (Logarithmic scale)
            max_views = df['Viewed_N'].max() if df['Viewed_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Viewed_N']) / np.log1p(max_views)
            
            # D. FINAL ATTRIBUTION SCORE
            df['Attribution_Score'] = (df['CTR_Decimal'] * 0.8) + (df['Log_Scale'] * 0.2)

            # --- DISPLAY ---
            st.subheader("🔍 Step 1: Raw Campaign Audit")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.subheader("🏆 Step 2: High-CTR Benchmarks (Ranked by Clicks/Viewed)")
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            
            # Formatted column for display
            winners['Final CTR %'] = winners['CTR_Percentage'].astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'Final CTR %', 'Attribution_Score']])

            if st.button("🚀 Step 3: Engineer Content from High-CTR DNA", key="engineer_btn_v2"):
                if not ACTIVE_KEY:
                    st.error("API Key missing! Add GEMINI_API_KEY to your Secrets.")
                else:
                    try:
                        # STABLE CONFIGURATION
                        genai.configure(api_key=ACTIVE_KEY)
                        # Remove 'models/' prefix - the SDK handles it
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        context_data = ""
                        for _, row in winners.iterrows():
                            context_data += f"Campaign ID: {row[id_col]} | CTR: {row['Final CTR %']} | Segment: {row.get(who_col)} | Message: {row[msg_col]}\n"

                        prompt = f"""
                        Analyze these High-CTR Winners for the segment: {seg_type} {sub_seg}.
                        TARGET GOAL: {intention}
                        WINNERS DATA: {context_data}
                        
                        TASK:
                        1. Factual Audit: Why did these Campaign IDs achieve high CTRs (Clicks/Viewed)?
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Table columns: Usage Rank, New Content, Reference ID, Source Segment, Hit %, Reasoning.
                        """
                        
                        with st.spinner("Analyzing high-performance DNA..."):
                            response = model.generate_content(prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as api_err:
                        if "429" in str(api_err):
                            st.error("⚠️ Daily API Quota Reached (20/day). Please try again tomorrow.")
                        elif "404" in str(api_err):
                            st.error("⚠️ Model Not Found. The API might be expecting 'gemini-1.5-pro' or 'gemini-1.5-flash'. Attempting to switch...")
                        else:
                            st.error(f"API Error: {api_err}")
        else:
            st.error("CSV Headers missing: Need 'Campaign ID', 'Message', 'Total Viewed(users)', and 'Total Clicked(users)'.")

    except Exception as e:
        st.error(f"Critical System Error: {e}")
else:
    st.info("👋 Upload your campaign CSVs to start the attribution engine.")    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin", key="sub_sidebar")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus", key="prod_base_sidebar")

    st.divider()
    
    st.header("⚙️ Ranking Logic")
    with st.expander("🔬 CTR Logic", expanded=True):
        st.markdown('<div class="logic-box"><b>Sure-Shot Formula:</b><br><span class="formula">CTR = (Clicks / Viewed) * 100</span><br><br><b>Weighting:</b><br>80% Engagement DNA<br>20% Volume Confidence</div>', unsafe_allow_html=True)
        st.caption("Focuses on messages that convert visual impressions into clicks.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 High-Performance Campaign Attribution")

# UNIQUE KEY to prevent StreamlitDuplicateElementId
uploaded_files = st.file_uploader("Upload Campaign CSVs", type="csv", accept_multiple_files=True, key="uploader_main")

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # UPDATED COLUMN MAPPING
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        view_col = col_map.get('total viewed(users)', None)
        clicks_col = col_map.get('total clicked(users)', None)

        if id_col and msg_col and view_col and clicks_col:
            # A. DATA CLEANING
            df['Viewed_N'] = pd.to_numeric(df[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # B. CALCULATE ACTUAL CTR PERCENTAGE (Clicks/Viewed)
            df['CTR_Decimal'] = (df['Clicks_N'] / df['Viewed_N']).replace([np.inf, -np.inf], 0).fillna(0)
            df['CTR_Percentage'] = (df['CTR_Decimal'] * 100).round(2)
            
            # C. VOLUME CONFIDENCE (Logarithmic scale)
            max_views = df['Viewed_N'].max() if df['Viewed_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Viewed_N']) / np.log1p(max_views)
            
            # D. FINAL ATTRIBUTION SCORE
            df['Attribution_Score'] = (df['CTR_Decimal'] * 0.8) + (df['Log_Scale'] * 0.2)

            # --- DISPLAY ---
            st.subheader("🔍 Step 1: Raw Campaign Audit")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.subheader("🏆 Step 2: High-CTR Benchmarks (Ranked by Clicks/Viewed)")
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            
            # Formatting for display
            winners['Final CTR %'] = winners['CTR_Percentage'].astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'Final CTR %', 'Attribution_Score']])

            if st.button("🚀 Step 3: Engineer Content from High-CTR DNA", key="engineer_btn"):
                if not ACTIVE_KEY:
                    st.error("API Key missing! Add GEMINI_API_KEY to your Secrets.")
                else:
                    try:
                        # Configure GenAI
                        genai.configure(api_key=ACTIVE_KEY)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        context_data = ""
                        for _, row in winners.iterrows():
                            context_data += f"Campaign ID: {row[id_col]} | CTR: {row['Final CTR %']} | Segment: {row.get(who_col)} | Message: {row[msg_col]}\n"

                        prompt = f"""
                        Analyze these High-CTR Winners for the segment: {seg_type} {sub_seg}.
                        TARGET GOAL: {intention}
                        WINNERS DATA: {context_data}
                        
                        TASK:
                        1. Factual Audit: Why did these Campaign IDs achieve high CTRs (Clicks/Viewed)?
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Table columns: Usage Rank, New Content, Reference ID, Source Segment, Hit %, Reasoning.
                        """
                        
                        with st.spinner("Analyzing high-performance DNA..."):
                            response = model.generate_content(prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as api_err:
                        if "429" in str(api_err):
                            st.error("⚠️ Daily API Quota Reached. Please try again tomorrow.")
                        else:
                            st.error(f"API Error: {api_err}")
        else:
            st.error("CSV Headers missing: Need 'Campaign ID', 'Message', 'Total Viewed(users)', and 'Total Clicked(users)'.")

    except Exception as e:
        st.error(f"Critical System Error: {e}")
else:
    st.info("👋 Upload your campaign CSVs to start the attribution engine.")
