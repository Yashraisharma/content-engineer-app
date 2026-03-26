import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Dual-Stream Edition", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 400px !important; }
    .formula-box { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 8px; 
        font-family: 'Courier New', monospace; 
        font-size: 0.95em;
        font-weight: bold;
        border-left: 6px solid #007bff;
        color: #1f2937;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & SEGMENTATION ---
with st.sidebar:
    st.title("🛡️ Dual-Stream Engineer")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale, Hygiene", key="side_kw")
    prod_description = st.text_area("Product Details", height=120, key="side_prod")
    intention = st.text_area("Primary Goal", height=120, key="side_goal")
    
    st.divider()
    
    st.header("🔍 Advanced Segmentation")
    seg_type = st.text_input("Segment Type", key="side_type")
    seg_reason = st.text_input("Reason for Segment", key="side_reason")
    sub_seg = st.text_input("Sub Segment", key="side_sub")
    spec_prod = st.text_input("Specific Product Base", key="side_base")

    st.divider()
    st.header("⚙️ System Logic")
    with st.expander("Logic 1: Performance Ranking", expanded=True):
        st.markdown('<div class="formula-box">Score = (CTR × 0.7) + (Vol × 0.3)</div>', unsafe_allow_html=True)
    with st.expander("Logic 2: Format Strategy", expanded=True):
        st.write("**7 Evolutionary + 3 Revolutionary**")
        st.caption("Replicating structural skeletons from File 2.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Growth Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

col_perf, col_format = st.columns(2)

# --- COLUMN 1: PERFORMANCE DATA (FILE 1) ---
with col_perf:
    st.subheader("📂 Stream 1: Performance DNA")
    perf_files = st.file_uploader("Upload Historical Performance CSVs", type="csv", accept_multiple_files=True, key="perf_up")
    
    if perf_files:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_perf = pd.concat(all_dfs, ignore_index=True)
        df_perf.columns = df_perf.columns.str.strip()
        
        # Mapping for Performance Logic
        c_map = {c.lower(): c for c in df_perf.columns}
        content_col = c_map.get('message', c_map.get('content', df_perf.columns[0]))
        ctr_col = c_map.get('ctr', None)
        imp_col = c_map.get('total viewed(users)', c_map.get('impression', c_map.get('sent', None)))

        if ctr_col and imp_col:
            df_perf['CTR_C'] = pd.to_numeric(df_perf[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df_perf['Imp_C'] = pd.to_numeric(df_perf[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            df_perf['Power_Score'] = (df_perf['CTR_C'] * 0.7) + ((df_perf['Imp_C'] / df_perf['Imp_C'].max()) * 0.3)
            winners_perf = df_perf.sort_values(by='Power_Score', ascending=False).head(10)
            st.dataframe(winners_perf[[content_col, ctr_col, imp_col, 'Power_Score']], use_container_width=True)
            
            if st.button("🚀 Analyze & Engineer Stream 1"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context = winners_perf[[content_col, ctr_col]].to_string(index=False)
                prompt = f"Using this Performance DNA:\n{context}\nGenerate 10 variations for {prod_description} using keywords {keywords_input} for segment {seg_type} {sub_seg}. Target: {intention}. Output Markdown Table."
                res = model.generate_content(prompt)
                st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

# --- COLUMN 2: FORMAT DATA (FILE 2) ---
with col_format:
    st.subheader("📂 Stream 2: Format Strategy")
    format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="format_up")
    
    if format_file:
        df_format = pd.read_csv(format_file)
        st.write("**Format Skeleton Detected:**")
        st.dataframe(df_format.head(5), use_container_width=True)
        
        if st.button("🚀 Run 10-Row Strategic Engineering (File 2 Style)"):
            if not ACTIVE_KEY:
                st.error("API Key Missing")
            else:
                with st.spinner("Engineering Style-Specific Variations..."):
                    genai.configure(api_key=ACTIVE_KEY)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    format_context = df_format.to_string(index=False)
                    prompt = f"""
                    REFERENCE FORMAT DATA: {format_context}
                    TASK: Generate 10 Content Rows. 7 Evolutionary (based on format skeleton), 3 Revolutionary.
                    Parameters: 
                    Product: {prod_description} 
                    Keywords: {keywords_input} 
                    Goal: {intention}
                    Segment: {seg_type} | {sub_seg} | {spec_prod}
                    
                    STRICT INSTRUCTION: Replicate the emoji usage, structural segmentation, and character style of the REFERENCE FORMAT DATA.
                    OUTPUT: Markdown table with: Usage Rank, New Content (File 2 Style), Segmentation Validation, Reference ID, Hit Percentage, Logic Hook.
                    """
                    
                    response = model.generate_content(prompt)
                    st.success("✅ Style Engineering Complete")
                    st.markdown(highlight_keywords(response.text, keywords_input), unsafe_allow_html=True)
