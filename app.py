import streamlit as st
import pandas as pd
from google import genai
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)
st.set_page_config(page_title="Attribution Engineer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 450px !important; }
    .attribution-card { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #007bff; font-size: 0.85em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & ARCHITECTURE ---
with st.sidebar:
    st.title("🛡️ Attribution Engineer Pro")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="Refill, Safety, Insulin")
    prod_description = st.text_area("Product/Offer Details", height=100)
    intention = st.text_area("Primary Goal", height=100)

    st.divider()
    
    st.header("🔍 Segment Anchors (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Chronic Lapsed")
    seg_reason = st.text_input("2. Reason", placeholder="e.g. 30-Day Inactive")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Insulin Users")
    spec_prod = st.text_input("4. Specific Product", placeholder="e.g. Lantus")

    st.divider()
    
    st.header("⚙️ Applied Engineering Logic")
    with st.expander("🔬 Relational Attribution", expanded=True):
        st.write("**Multi-Column Ranking:**")
        st.markdown('<div class="attribution-card"><span class="formula">Score = (CTR*0.5) + (Conv%*0.3) + (Volume*0.2)</span></div>', unsafe_allow_html=True)
        st.caption("Matches Campaign ID + Who Query + Message DNA to identify 'Sure Shot' patterns.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Deep-Column Campaign Analysis & Attribution")

uploaded_files = st.file_uploader("Upload Full-Header Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # REQUIRED COLUMN MAPPING (Based on your high-fidelity list)
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        ctr_col = col_map.get('total ctr', None)
        conv_col = col_map.get('click through conversions %', None)
        sent_col = col_map.get('total sent(users)', None)

        if all([id_col, msg_col, ctr_col]):
            # Data Cleaning & Conversion
            df['CTR_N'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Conv_N'] = pd.to_numeric(df[conv_col].astype(str).str.replace('%', ''), errors='coerce') / 100 if conv_col else 0
            df['Sent_N'] = pd.to_numeric(df[sent_col].astype(str).str.replace(',', ''), errors='coerce') if sent_col else 0
            
            # MULTI-DIMENSIONAL RANKING LOGIC
            df['Attribution_Score'] = (df['CTR_N'] * 0.5) + (df['Conv_N'] * 0.3) + ((df['Sent_N'] / df['Sent_N'].max()) * 0.2)
            
            # Segment Relevance Weighting
            target_str = f"{seg_type} {sub_seg} {spec_prod}".lower()
            if who_col and target_str.strip():
                df['Relevance'] = df[who_col].apply(lambda x: 1.5 if any(word in str(x).lower() for word in target_str.split()) else 1.0)
                df['Final_Score'] = df['Attribution_Score'] * df['Relevance']
            else:
                df['Final_Score'] = df['Attribution_Score']

            winners = df.sort_values(by='Final_Score', ascending=False).head(10).copy()

            # UI: Performance References with Campaign ID and Segment mapping
            st.subheader("🏆 Attribution Benchmarks (Ranked Winners)")
            st.dataframe(winners[[id_col, msg_col, who_col, ctr_col, 'Final_Score']], use_container_width=True)

            if st.button("🚀 Engineer 10 Attribution-Mapped Variations"):
                client = genai.Client(api_key=ACTIVE_KEY)
                
                # Context building with FULL metadata
                context = ""
                for _, row in winners.iterrows():
                    context += f"""
                    ID: {row[id_col]}
                    SEGMENT: {row.get(who_col, 'N/A')}
                    MESSAGE: {row[msg_col]}
                    METRICS: CTR={row.get(ctr_col)}, Conv={row.get(conv_col, '0%')}
                    -----------------------------------
                    """

                prompt = f"""
                You are a Strategic Growth Analyst. 
                TARGET: {seg_type} | {seg_reason} | {sub_seg} | {spec_prod}
                PRODUCT: {prod_description} | GOAL: {intention} | KEYWORDS: {keywords_input}
                
                HISTORICAL PERFORMANCE ARCHITECTURE:
                {context}
                
                TASK:
                1. Perform a 'Segment-Message Audit': Explain which Campaign IDs succeeded for specific Who Query segments and why.
                2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                3. EVERY suggestion must list the 'Reference Campaign ID' it was modeled after.
                
                OUTPUT: Markdown table with: Usage Rank, New Content, Reference Campaign ID, Source Segment (Who Query), Hit %, Reasoning.
                """
                
                with st.spinner("Processing deep-column attribution..."):
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering Complete")
                    st.markdown(response.text)

    except Exception as e:
        st.error(f"Critical Error in Data Processing: {e}")
else:
    st.info("👋 Upload your campaign CSVs to map Campaign IDs, Segments, and Message DNA.")
