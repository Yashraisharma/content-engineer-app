import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Vertical Edition", layout="wide")

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
        margin-bottom: 10px;
    }
    .stream-header {
        background-color: #1f2937;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin-top: 25px;
        font-weight: bold;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale", key="v8_kw")
    prod_description = st.text_area("Product Details", height=100, key="v8_prod")
    intention = st.text_area("Primary Goal", height=100, key="v8_goal")
    
    st.divider()
    
    st.header("🔍 Advanced Segmentation")
    seg_type = st.text_input("Segment Type", key="v8_type")
    seg_reason = st.text_input("Reason for Segment", key="v8_reason")
    sub_seg = st.text_input("Sub Segment", key="v8_sub")
    spec_prod = st.text_input("Specific Product Base", key="v8_base")

    st.divider()
    st.header("⚙️ Applied Logic")
    st.markdown('<div class="formula-box">Stream 1 (Perf):<br>Score = (CTR × 0.7) + (Vol × 0.3)</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula-box">Stream 2 (Style):<br>Replicate Format Skeleton (7+3)</div>', unsafe_allow_html=True)

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Growth Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE DNA ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance-Based Analysis (Historical Winners)</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Historical Performance CSVs", type="csv", accept_multiple_files=True, key="v8_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        # SUPER-MAPPER: Finding columns regardless of exact naming
        cols = [c.lower() for c in df_p.columns]
        
        # 1. Find Message/Content
        msg_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['message', 'content', 'text', 'body'])), 0)
        content_col = df_p.columns[msg_idx]
        
        # 2. Find CTR
        ctr_idx = next((i for i, c in enumerate(cols) if 'ctr' in c), None)
        
        # 3. Find Volume (Impression/Sent/Viewed)
        vol_idx = next((i for i, i_c in enumerate(cols) if any(x in i_c for x in ['viewed', 'impression', 'sent', 'delivered', 'reach'])), None)

        if ctr_idx is not None and vol_idx is not None:
            ctr_col = df_p.columns[ctr_idx]
            vol_col = df_p.columns[vol_idx]
            
            # Clean Data
            df_p['CTR_Clean'] = pd.to_numeric(df_p[ctr_col].astype(str).str.replace('%', ''), errors='coerce').fillna(0) / 100
            df_p['Vol_Clean'] = pd.to_numeric(df_p[vol_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Power Score
            df_p['Power_Score'] = (df_p['CTR_Clean'] * 0.7) + ((df_p['Vol_Clean'] / (df_p['Vol_Clean'].max() if df_p['Vol_Clean'].max() > 0 else 1)) * 0.3)
            
            winners_p = df_p.sort_values(by='Power_Score', ascending=False).head(10).copy()
            
            st.write("### 🏆 Top 10 Performance Benchmarks")
            st.dataframe(winners_p[[content_col, ctr_col, vol_col, 'Power_Score']], use_container_width=True)
            
            if st.button("🚀 Run Stream 1: Performance Engineering", key="v8_btn_p"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = winners_p[[content_col, ctr_col]].to_string(index=False)
                prompt_p = f"Using Performance DNA:\n{context_p}\nEngineer 10 variations for {prod_description} using keywords {keywords_input} for segment {seg_type}. Target: {intention}."
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Column Mapping Failed. Ensure your CSV has columns for 'CTR' and 'Viewed/Sent'.")
            st.write("Columns found:", list(df_p.columns))
    except Exception as e:
        st.error(f"Stream 1 Error: {e}")

st.divider()

# --- STREAM 2: FORMAT STRATEGY ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format-Based Engineering (Style Replication)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v8_format_up")

if format_file:
    try:
        df_f = pd.read_csv(format_file)
        st.write("### 📝 Detected Style Skeletons")
        st.dataframe(df_f.head(5), use_container_width=True)
        
        if st.button("🚀 Run Stream 2: Strategic Format Engineering", key="v8_btn_f"):
            if not ACTIVE_KEY:
                st.error("API Key Missing")
            else:
                with st.spinner("Engineering Style-Specific Variations..."):
                    genai.configure(api_key=ACTIVE_KEY)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    format_context = df_f.to_string(index=False)
                    
                    prompt_f = f"""
                    REFERENCE FORMAT DATA: {format_context}
                    TASK: Generate 10 Content Rows. 7 Evolutionary (based on format skeleton), 3 Revolutionary.
                    Parameters: 
                    Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                    Segment: {seg_type} | {sub_seg} | {seg_reason}
                    
                    STRICT INSTRUCTION: Replicate the emoji usage, structural segmentation, and character style.
                    OUTPUT: Markdown table with columns: Usage Rank, New Content (File 2 Style), Segmentation Validation, Reference ID, Hit Percentage, Logic Hook.
                    """
                    
                    response_f = model.generate_content(prompt_f)
                    st.success("✅ Style Engineering Complete")
                    st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Stream 2 Error: {e}")
# --- 2. SIDEBAR: PARAMETERS & SEGMENTATION ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale, Hygiene", key="v7_kw")
    prod_description = st.text_area("Product Details", height=100, key="v7_prod")
    intention = st.text_area("Primary Goal", height=100, key="v7_goal")
    
    st.divider()
    
    st.header("🔍 Advanced Segmentation")
    seg_type = st.text_input("Segment Type", key="v7_type")
    seg_reason = st.text_input("Reason for Segment", key="v7_reason")
    sub_seg = st.text_input("Sub Segment", key="v7_sub")
    spec_prod = st.text_input("Specific Product Base", key="v7_base")

    st.divider()
    st.header("⚙️ Applied Logic")
    st.markdown('<div class="formula-box">Stream 1: (CTR × 0.7) + (Vol × 0.3)</div>', unsafe_allow_html=True)
    st.markdown('<div class="formula-box">Stream 2: Format Replication (7+3)</div>', unsafe_allow_html=True)

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Growth Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE DNA ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance-Based Engineering (Historical CTR)</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Historical Performance CSVs", type="csv", accept_multiple_files=True, key="v7_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        c_map = {c.lower(): c for c in df_p.columns}
        content_col = c_map.get('message', c_map.get('content', df_p.columns[0]))
        ctr_col = c_map.get('ctr', None)
        imp_col = c_map.get('total viewed(users)', c_map.get('impression', c_map.get('sent', None)))

        if ctr_col and imp_col:
            df_p['CTR_C'] = pd.to_numeric(df_p[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df_p['Imp_C'] = pd.to_numeric(df_p[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            df_p['Power_Score'] = (df_p['CTR_C'] * 0.7) + ((df_p['Imp_C'] / df_p['Imp_C'].max()) * 0.3)
            winners_p = df_p.sort_values(by='Power_Score', ascending=False).head(10).copy()
            
            st.write("### 🏆 Performance Benchmarks")
            st.dataframe(winners_p[[content_col, ctr_col, imp_col, 'Power_Score']], use_container_width=True)
            
            if st.button("🚀 Run Stream 1: Performance DNA Analysis", key="v7_btn_p"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = winners_p[[content_col, ctr_col]].to_string(index=False)
                prompt_p = f"Using Performance DNA:\n{context_p}\nEngineer 10 variations for {prod_description} using keywords {keywords_input} for segment {seg_type} {sub_seg}. Target: {intention}."
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Stream 1 Error: {e}")

st.divider()

# --- STREAM 2: FORMAT STRATEGY ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format-Based Engineering (Style Replication)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v7_format_up")

if format_file:
    try:
        df_f = pd.read_csv(format_file)
        st.write("### 📝 Detected Style Skeletons")
        st.dataframe(df_f.head(5), use_container_width=True)
        
        if st.button("🚀 Run Stream 2: Strategic Format Engineering", key="v7_btn_f"):
            if not ACTIVE_KEY:
                st.error("API Key Missing")
            else:
                with st.spinner("Engineering Style-Specific Variations..."):
                    genai.configure(api_key=ACTIVE_KEY)
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    format_context = df_f.to_string(index=False)
                    
                    # Using your Strategic Edition Logic
                    prompt_f = f"""
                    REFERENCE FORMAT DATA: {format_context}
                    TASK: Generate 10 Content Rows. 7 Evolutionary (based on format skeleton), 3 Revolutionary.
                    Parameters: 
                    Product: {prod_description} 
                    Keywords: {keywords_input} 
                    Goal: {intention}
                    Segment: {seg_type} | {sub_seg} | {seg_reason}
                    
                    STRICT INSTRUCTION: Replicate the emoji usage, structural segmentation (e.g., Content/Segmentation/Impression lines), and character style.
                    OUTPUT: Markdown table with columns: Usage Rank, New Content (File 2 Style), Segmentation Validation, Reference ID, Hit Percentage, Logic Hook.
                    """
                    
                    response_f = model.generate_content(prompt_f)
                    st.success("✅ Style Engineering Complete")
                    st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
                    st.download_button("📥 Download Stream 2 Analysis", response_f.text, file_name="engineered_formats.txt")
    except Exception as e:
        st.error(f"Stream 2 Error: {e}")
