import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | CTR Fix", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stream-header { background-color: #1f2937; color: white; padding: 15px; border-radius: 5px; margin-top: 25px; font-weight: bold; font-size: 1.2em; }
    .highlight-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #ff4b4b; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    keywords_input = st.text_input("Target Keywords", key="v10_kw")
    prod_description = st.text_area("Product Details", height=100, key="v10_prod")
    intention = st.text_area("Primary Goal", height=100, key="v10_goal")
    st.divider()
    st.header("🔍 Segmentation")
    seg_type = st.text_input("Segment Type", key="v10_type")
    sub_seg = st.text_input("Sub Segment", key="v10_sub")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Growth Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance Analysis</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Performance CSVs", type="csv", accept_multiple_files=True, key="v10_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        cols = [c.lower() for c in df_p.columns]
        
        # Mapping Logic
        msg_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['message', 'content', 'text'])), 0)
        ctr_idx = next((i for i, c in enumerate(cols) if 'ctr' in c), None)
        vol_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['viewed', 'sent', 'impression', 'delivered'])), None)

        if ctr_idx is not None and vol_idx is not None:
            content_col = df_p.columns[msg_idx]
            orig_ctr_col = df_p.columns[ctr_idx]
            vol_col = df_p.columns[vol_idx]
            
            # --- HARD-CLEANING CTR ---
            def clean_ctr(val):
                try:
                    s = str(val).replace('%', '').replace(',', '').strip()
                    num = float(s)
                    # If it's a decimal like 0.05, convert to 5.0
                    if 0 < num < 1:
                        return num * 100
                    return num
                except:
                    return 0.0

            df_p['CTR_Final'] = df_p[orig_ctr_col].apply(clean_ctr)
            df_p['Vol_Num'] = pd.to_numeric(df_p[vol_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Power Score Calculation
            df_p['Power_Score'] = (df_p['CTR_Final'] * 0.7) + ((df_p['Vol_Num'] / (df_p['Vol_Num'].max() or 1)) * 30)
            
            winners_p = df_p.sort_values(by='Power_Score', ascending=False).head(10).copy()
            
            # Formatted Display Column
            winners_p['CTR %'] = winners_p['CTR_Final'].apply(lambda x: f"{x:.2f}%")
            
            st.write("### 🏆 Top 10 Performance Benchmarks")
            # Using st.table for clearer numeric visibility
            st.table(winners_p[[content_col, 'CTR %', vol_col, 'Power_Score']])
            
            if st.button("🚀 Run Stream 1 Engineering"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = winners_p[[content_col, 'CTR %']].to_string(index=False)
                prompt_p = f"Using Performance DNA:\n{context_p}\nEngineer 10 variations for {prod_description} with keywords {keywords_input} for segment {seg_type}. Goal: {intention}."
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
        else:
            st.error(f"❌ Could not find CTR or Volume columns. Found: {list(df_p.columns)}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# --- STREAM 2: FORMAT ---
st.markdown('<div class="stream-header">📂 STREAM 2: Style Strategy</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v10_format_up")

if format_file:
    df_f = pd.read_csv(format_file)
    st.dataframe(df_f.head(3), use_container_width=True)
    
    if st.button("🚀 Run Stream 2 (File 2 Style)"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        format_context = df_f.to_string(index=False)
        prompt_f = f"""
        FORMAT: {format_context}
        TASK: 10 variations (7 Evo, 3 Revo). 
        STRICT: Replicate emoji/segment style (e.g. hygiene/BOGO format).
        PARAMS: {prod_description} | {keywords_input} | {seg_type} {sub_seg}
        """
        response_f = model.generate_content(prompt_f)
        st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
