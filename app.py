import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Sure-Shot Growth Engineer", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #28a745; color: white; font-weight: bold; border: none; }
    section[data-testid="stSidebar"] { width: 450px !important; }
    .logic-box { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #28a745; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: THE 7 CRITICAL INPUTS ---
with st.sidebar:
    st.title("🎯 Control Center")
    
    with st.expander("📝 1. Core Content DNA", expanded=True):
        keywords_input = st.text_input("Core Keywords", placeholder="Refill, Safety, Insulin", key="v6_kw")
        prod_description = st.text_area("Product Details", height=100, placeholder="Describe the item...", key="v6_prod")
        intention = st.text_area("Primary Goal/Intention", height=100, placeholder="e.g. Increase repeat orders", key="v6_goal")

    with st.expander("🔍 2. Advanced Segment Targeting", expanded=True):
        seg_type = st.text_input("Segment Type", placeholder="e.g. Lapsed Users", key="v6_type")
        seg_reason = st.text_input("Reason for Segment", placeholder="e.g. 30-Day Inactive", key="v6_reason")
        sub_seg = st.text_input("Sub Segment", placeholder="e.g. Chronic/Insulin", key="v6_sub")
        spec_prod = st.text_input("Specific Product Base", placeholder="e.g. Lantus", key="v6_base")

    st.divider()
    
    st.header("⚙️ Ranking Logic")
    with st.expander("🔬 Segment-First Formula", expanded=True):
        st.markdown('''
        <div class="logic-box">
        <b>Sure-Shot Score:</b><br>
        <span class="formula">(CTR * 0.5) + (SegMatch * 0.4) + (LogScale * 0.1)</span>
        </div>
        ''', unsafe_allow_html=True)
        st.caption("Matches your 'Who Query' against the 4 Segment boxes for high-relevance cloning.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Segment & Format Engineering Engine")

col1, col2 = st.columns(2)

with col1:
    perf_files = st.file_uploader("📂 Step A: Upload Performance CSVs (CTR Data)", type="csv", accept_multiple_files=True, key="v6_perf")

with col2:
    format_file = st.file_uploader("📂 Step B: Upload Format Template (Examples)", type="csv", key="v6_format", help="Upload a file with your favorite message formats to replicate.")

if perf_files:
    try:
        # Load Performance Data
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # Load Format Template if exists
        format_dna = ""
        if format_file:
            f_df = pd.read_csv(format_file)
            format_dna = f_df.to_string(index=False)
            st.success("✅ Format DNA Loaded: AI will mimic the emoji style and structure.")

        # Column Mapping
        col_map = {c.lower(): c for c in df.columns}
        id_col, msg_col = col_map.get('campaign id'), col_map.get('message')
        who_col = col_map.get('who query')
        view_col, clicks_col = col_map.get('total viewed(users)'), col_map.get('total clicked(users)')

        if all([id_col, msg_col, view_col, clicks_col]):
            # A. CALCULATE CTR (Clicks / Viewed)
            df['Viewed_N'] = pd.to_numeric(df[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['CTR_Perc'] = (df['Clicks_N'] / df['Viewed_N']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # B. SEGMENT MATCHING (40% Weight)
            # We look for the user's sidebar inputs inside the "Who Query" column
            target_str = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower()
            if who_col and target_str.strip():
                df['Seg_Match'] = df[who_col].apply(lambda x: 1.0 if any(word in str(x).lower() for word in target_str.split() if word) else 0.0)
            else:
                df['Seg_Match'] = 0.5

            # C. FINAL SURE-SHOT RANKING
            max_v = df['Viewed_N'].max() if df['Viewed_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Viewed_N']) / np.log1p(max_v)
            df['Final_Score'] = (df['CTR_Perc'] * 0.5) + (df['Seg_Match'] * 0.4) + (df['Log_Scale'] * 0.1)

            # --- DISPLAY ---
            st.subheader("🏆 High-Performance Segment Benchmarks")
            winners = df.sort_values(by='Final_Score', ascending=False).head(10).copy()
            winners['CTR %'] = (winners['CTR_Perc'] * 100).round(2).astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'CTR %', 'Final_Score']])

            if st.button("🚀 Step C: Engineer 10 Segment-Specific Variations"):
                if not ACTIVE_KEY:
                    st.error("API Key Missing!")
                else:
                    try:
                        genai.configure(api_key=ACTIVE_KEY)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        historical_dna = ""
                        for _, row in winners.iterrows():
                            historical_dna += f"ID: {row[id_col]} | CTR: {row['CTR %']} | Message: {row[msg_col]}\n"

                        prompt = f"""
                        You are a Growth Content Engineer.
                        
                        NEW PRODUCT: {prod_description}
                        TARGET GOAL: {intention}
                        KEYWORDS: {keywords_input}
                        TARGET SEGMENT: {seg_type} ({seg_reason})
                        
                        HISTORICAL PERFORMANCE DNA:
                        {historical_dna}
                        
                        REPLICATION FORMAT EXAMPLES:
                        {format_dna}
                        
                        TASK:
                        1. Factual Audit: Briefly explain why these historical messages worked for this specific segment.
                        2. Engineer 10 Variations: 7 Evolutionary (based on winners), 3 Revolutionary (new hooks).
                        3. Use Emojis and the exact structural style found in the 'Replication Format Examples'.
                        4. Output as a Markdown Table: [Variation # | Content | Ref ID | Segment Logic | Expected Impact]
                        """
                        
                        with st.spinner("Engineering high-reactivity content..."):
                            response = model.generate_content(prompt)
                            st.success("✅ Content Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as api_err:
                        st.error(f"API Error: {api_err}")
        else:
            st.error("Header Error: Ensure CSV has 'Campaign ID', 'Message', 'Total Viewed(users)', and 'Total Clicked(users)'.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Upload Performance Data and Format Templates to begin.")
