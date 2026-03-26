import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Strategic Suite", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 400px !important; }
    .formula-box { 
        background-color: #f0f2f6; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.95em; font-weight: bold; border-left: 6px solid #10b981; color: #1f2937; margin-bottom: 10px;
    }
    .stream-header {
        background-color: #1f2937; color: white; padding: 15px; border-radius: 5px; margin-top: 25px; font-weight: bold; font-size: 1.2em;
    }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & PROJECT LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale", key="final_kw")
    prod_description = st.text_area("Product Details", height=100, key="final_prod")
    intention = st.text_area("Primary Goal", height=100, key="final_goal")
    
    st.divider()
    st.header("🔍 Advanced Segmentation")
    seg_type = st.text_input("Segment Type", key="final_seg")
    sub_seg = st.text_input("Sub Segment", key="final_sub")
    spec_prod = st.text_input("Specific Product Base", key="final_base")

    st.divider()
    st.header("📋 Project Summary")
    st.write("Extracts proven success DNA from raw metrics to engineer 10 optimized variations. Prioritizes scale-validated winners.")
    
    st.header("⚙️ Applied Logic")
    with st.expander("1. Scale-Validation Engine", expanded=True):
        st.markdown('<div class="formula-box">CTR% = (Clicks / Viewed or IMP) × 100</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">Score = CTR% * (Vol / (Vol + Avg_Vol))</div>', unsafe_allow_html=True)
        st.caption("Ensures 15k-view winners rank above 2-view flukes.")

    with st.expander("2. 7+3 Strategic Engineering", expanded=True):
        st.write("**7 Evolutionary:** Performance-led structural swaps.")
        st.write("**3 Revolutionary:** Psychological angle pivots.")

# --- 3. CORE PROCESSING ENGINE ---
def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

def process_and_rank(df, stream_label):
    df.columns = df.columns.str.strip()
    cols_low = [c.lower() for c in df.columns]
    
    # Universal column detection for Viewed/IMP and Clicks
    msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
    view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'impression', 'sent'])), None)
    click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'clicks'])), None)
    
    if view_idx is not None and click_idx is not None:
        content_col, v_col, c_col = df.columns[msg_idx], df.columns[view_idx], df.columns[click_idx]
        
        # Clean Metrics
        df['V_N'] = pd.to_numeric(df[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['C_N'] = pd.to_numeric(df[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Mandated CTR Calculation
        df['True_CTR'] = (df['C_N'] / df['V_N'].replace(0, np.nan)) * 100
        df['True_CTR'] = df['True_CTR'].fillna(0.0)
        
        # Confidence-Weighted Score (Rewards Scale, Punishes Flukes)
        avg_v = df['V_N'].mean()
        df['Confidence_Score'] = df['True_CTR'] * (df['V_N'] / (df['V_N'] + avg_v))
        
        # Ranking
        ranked = df.sort_values(by='Confidence_Score', ascending=False)
        ranked['CTR% Display'] = ranked['True_CTR'].apply(lambda x: f"{x:.2f}%")
        
        # Output UI
        st.write(f"### 📑 Full {stream_label} Ranked Dataset")
        st.dataframe(ranked[[content_col, 'CTR% Display', v_col, c_col, 'Confidence_Score']], use_container_width=True)
        
        st.write(f"### 🏆 Top 10 High-Scale Winners")
        st.table(ranked.head(10)[[content_col, 'CTR% Display', v_col, 'Confidence_Score']])
        
        return ranked, content_col
    else:
        st.error(f"Could not find required metrics in {stream_label}. Check for 'Viewed/IMP' and 'Clicked' columns.")
        return None, None

# --- 4. MAIN DASHBOARD ---
st.title("📊 Unified Strategic Engineering Dashboard")

# --- STREAM 1 ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance Analysis (Historical Winners)</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 Performance CSVs", type="csv", accept_multiple_files=True, key="final_s1_up")

if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, content_col_s1 = process_and_rank(df_s1, "Stream 1")
    
    if ranked_s1 is not None and st.button("🚀 Run Stream 1 Strategic Engineering", key="btn_s1"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s1.head(10)[[content_col_s1, 'CTR% Display']].to_string(index=False)
        prompt = f"TASK: 10 variations (7 Evo, 3 Revo) using DNA:\n{context}\nParams: {prod_description} | Keywords: {keywords_input} | Segment: {seg_type}"
        res = model.generate_content(prompt)
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# --- STREAM 2 ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format Strategy (Style Replication)</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format Template CSV", type="csv", key="final_s2_up")

if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, content_col_s2 = process_and_rank(df_s2, "Stream 2")
    
    if ranked_s2 is not None and st.button("🚀 Run Stream 2 Strategic Engineering", key="btn_s2"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s2.head(10)[[content_col_s2, 'CTR% Display']].to_string(index=False)
        prompt = f"""
        FORMAT & DNA: {context}
        TASK: 10 Content Rows. 7 Evolutionary, 3 Revolutionary.
        STRICT: Replicate structural segmentation (BOGO/Hygiene), emoji usage, and style.
        Params: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
        """
        res = model.generate_content(prompt)
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
