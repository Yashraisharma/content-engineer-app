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
    .logic-header { background-color: #e9ecef; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 0.9em; margin-bottom: 15px; border-left: 5px solid #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & OPTIONAL BOXES ---
with st.sidebar:
    st.title("🛡️ Attribution Engineer Pro")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="e.g. Refill, Insulin, Safety")
    prod_description = st.text_area("Product Details", height=150)
    intention = st.text_area("Primary Goal", height=150)

    st.divider()
    
    st.header("🔍 Advanced Targeting (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed Users")
    seg_reason = st.text_input("2. Reason for Segment", placeholder="e.g. 30-Day Inactive")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus")

    st.divider()
    
    st.header("⚙️ Engineering Logic")
    with st.expander("🔬 Multi-Metric Attribution", expanded=True):
        st.write("**Ranking Formula:**")
        st.code("Score = (CTR*0.4) + (Conv%*0.4) + (Rel_Rev*0.2)")
        st.caption("Matches Campaign ID + Who Query + Message DNA to identify 'Sure Shot' patterns.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Deep-Column Campaign Analysis & Attribution")

uploaded_files = st.file_uploader("Upload Full-Header Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # IDENTIFYING COLUMNS FROM YOUR SPECIFIC LIST
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        ctr_col = col_map.get('total ctr', None)
        conv_col = col_map.get('click through conversions %', None)
        rev_col = col_map.get('click through conversion revenue', None)
        sent_col = col_map.get('total sent(users)', None)

        if id_col and msg_col and ctr_col:
            # Data Cleaning for Metrics
            df['CTR_N'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Conv_N'] = pd.to_numeric(df[conv_col].astype(str).str.replace('%', ''), errors='coerce') / 100 if conv_col else 0
            df['Rev_N'] = pd.to_numeric(df[rev_col], errors='coerce').fillna(0)
            
            # CORE RANKING LOGIC
            df['Base_Score'] = (df['CTR_N'] * 0.4) + (df['Conv_N'] * 0.4) + ((df['Rev_N'] / (df['Rev_N'].max() + 1)) * 0.2)
            
            # Applying Optional Segment Boost (1.5x)
            target_str = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower()
            if who_col and target_str.strip():
                df['Relevance'] = df[who_col].apply(lambda x: 1.5 if any(word in str(x).lower() for word in target_str.split()) else 1.0)
                df['Attribution_Score'] = df['Base_Score'] * df['Relevance']
            else:
                df['Attribution_Score'] = df['Base_Score']

            # --- DISPLAY SECTION ---
            st.markdown("### 🔍 Step 1: Raw Campaign Audit (All Columns)")
            st.write("Review the complete data source for all campaigns before ranking.")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.markdown("### 🏆 Step 2: Performance Ranking & Metric Analysis")
            st.write("The logic has identified these 10 campaigns as the 'DNA Blueprints' for your current goal.")
            
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            
            # Displaying winners with Campaign ID, Segment, and the critical Metrics
            display_cols = [id_col, who_col, msg_col, ctr_col]
            if conv_col: display_cols.append(conv_col)
            display_cols.append('Attribution_Score')
            
            st.table(winners[display_cols])

            if st.button("🚀 Step 3: Engineer 10 Attribution-Mapped Variations"):
                client = genai.Client(api_key=ACTIVE_KEY)
                
                # Context building with FULL metadata for AI reasoning
                context = ""
                for _, row in winners.iterrows():
                    context += f"""
                    Campaign ID: {row[id_col]} | Segment (Who Query): {row.get(who_col, 'N/A')}
                    Message: {row[msg_col]}
                    Performance: CTR={row.get(ctr_col)}, Conv%={row.get(conv_col, '0%')}, Score={row['Attribution_Score']:.4f}
                    -----------------------------------
                    """

                prompt = f"""
                You are a Senior Strategic Growth Analyst.
                TARGET SEGMENT: {seg_type} | {seg_reason} | {sub_seg} | {spec_prod}
                PRODUCT: {prod_description} | GOAL: {intention} | KEYWORDS: {keywords_input}
                
                HISTORICAL BENCHMARKS:
                {context}
                
                TASK:
                1. 'Factual Segment Audit': List the Campaign IDs that performed best and explain WHY they worked for their 'Who Query'.
                2. Engineer 10 Variations (7 Evolutionary based on the best Campaign IDs, 3 Revolutionary for new friction removal).
                3. Every row must show the 'Reference Campaign ID' and 'Source Segment'.
                
                OUTPUT: Markdown table with: Usage Rank, New Content, Reference Campaign ID, Source Segment (Who Query), Hit %, Reactivity Reasoning.
                """
                
                with st.spinner("Analyzing deep-column attribution..."):
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering & Analysis Complete")
                    st.markdown(response.text)
        else:
            st.error("Required columns (Campaign ID, Message, Total CTR) not found. Please check CSV headers.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Upload your campaign CSVs to map Campaign IDs, Who Queries, and Performance DNA.")    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Chronic/Insulin Users")
    spec_prod = st.text_input("4. Specific Product Base", placeholder="e.g. Lantus, Humalog")

    st.divider()
    
    # PROJECT SUMMARY & LOGIC
    st.header("⚙️ Applied Engineering Logic")
    
    with st.expander("🔬 Full-Spectrum Attribution", expanded=True):
        st.write("**Multi-Column Ranking:**")
        st.markdown('<div class="attribution-card"><span class="formula">Score = (CTR*0.4) + (Conv%*0.4) + (Revenue*0.2)</span></div>', unsafe_allow_html=True)
        st.caption("Matches Campaign ID + Who Query + Message DNA to identify 'Sure Shot' patterns.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Deep-Column Campaign Analysis & Attribution")

uploaded_files = st.file_uploader("Upload Full-Header Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        # MAPPING EVERY HEADER IN YOUR LIST
        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        ctr_col = col_map.get('total ctr', None)
        conv_col = col_map.get('click through conversions %', None)
        rev_col = col_map.get('click through conversion revenue', None)
        sent_col = col_map.get('total sent(users)', None)

        if all([id_col, msg_col, ctr_col]):
            # Data Cleaning
            df['CTR_N'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Conv_N'] = pd.to_numeric(df[conv_col].astype(str).str.replace('%', ''), errors='coerce') / 100 if conv_col else 0
            df['Rev_N'] = pd.to_numeric(df[rev_col], errors='coerce').fillna(0)
            df['Sent_N'] = pd.to_numeric(df[sent_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # FULL-SPECTRUM RANKING (Logic prioritizes Performance + Revenue + Reach)
            df['Attribution_Score'] = (df['CTR_N'] * 0.4) + (df['Conv_N'] * 0.4) + ((df['Rev_N'] / (df['Rev_N'].max() + 1)) * 0.2)
            
            # Segment Relevance Logic
            target_str = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower()
            if who_col and target_str.strip():
                df['Relevance'] = df[who_col].apply(lambda x: 1.5 if any(word in str(x).lower() for word in target_str.split()) else 1.0)
                df['Final_Score'] = df['Attribution_Score'] * df['Relevance']
            else:
                df['Final_Score'] = df['Attribution_Score']

            winners = df.sort_values(by='Final_Score', ascending=False).head(10).copy()

            # UI: Performance References
            st.subheader("🏆 Winning Segment Analysis (Ranked by ID & Who Query)")
            st.dataframe(winners[[id_col, who_col, msg_col, ctr_col, 'Final_Score']], use_container_width=True)

            if st.button("🚀 Engineer 10 Attribution-Mapped Variations"):
                client = genai.Client(api_key=ACTIVE_KEY)
                
                # Context building with FULL metadata for the AI Audit
                context = ""
                for _, row in winners.iterrows():
                    context += f"""
                    Campaign ID: {row[id_col]}
                    Target Segment (Who Query): {row.get(who_col, 'N/A')}
                    Message DNA: {row[msg_col]}
                    Metrics: CTR={row.get(ctr_col)}, Conv={row.get(conv_col, '0%')}, Rev={row.get(rev_col, 0)}
                    -----------------------------------
                    """

                prompt = f"""
                You are a Senior Strategic Growth Analyst.
                
                CURRENT TARGET: 
                - Segment: {seg_type}
                - Reason: {seg_reason}
                - Sub-Segment: {sub_seg}
                - Product Base: {spec_prod}
                - Product Details: {prod_description}
                - Primary Goal: {intention}
                
                HISTORICAL PERFORMANCE ARCHITECTURE:
                {context}
                
                TASK:
                1. 'Factual Segment Audit': Explain which Campaign IDs over-indexed for specific 'Who Queries' and estimate the behavioral reason.
                2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                3. EVERY suggestion MUST list the 'Reference Campaign ID' and its associated 'Who Query'.
                
                OUTPUT: Markdown table with: Usage Rank, New Content, Reference Campaign ID, Source Segment (Who Query), Hit %, Reactivity Reasoning.
                """
                
                with st.spinner("Analyzing deep-column attribution..."):
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering & Analysis Complete")
                    st.markdown(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Upload your campaign CSVs to map Campaign IDs, Who Queries, and Performance DNA.")
