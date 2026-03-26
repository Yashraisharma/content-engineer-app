import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Unified Metrics", layout="wide")

st.markdown("""
    <style>
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

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Keywords", placeholder="e.g. BOGO, Sale", key="v22_kw")
    prod_description = st.text_area("Product Details", height=100, key="v22_prod")
    seg_type = st.text_input("Segment Type", key="v22_type")

    st.divider()
    st.header("⚙️ Unified Logic")
    with st.expander("Confidence Scale Engine", expanded=True):
        st.markdown('<div class="formula-box">CTR% = (Clicks / Viewed or IMP) × 100</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">Score = CTR% * (Vol / (Vol + Avg_Vol))</div>', unsafe_allow_html=True)
        st.caption("Validates performance against impression scale to find true winners.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Unified Metric Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE ---
st.markdown('<div class="stream-header">📂 STREAM 1: Historical Performance Analysis</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Stream 1 CSVs", type="csv", accept_multiple_files=True, key="v22_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        cols_low = [c.lower() for c in df_p.columns]
        msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
        view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'impression'])), None)
        click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'clicks'])), None)

        if view_idx is not None and click_idx is not None:
            content_col, v_col, c_col = df_p.columns[msg_idx], df_p.columns[view_idx], df_p.columns[click_idx]
            
            df_p['V_N'] = pd.to_numeric(df_p[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['C_N'] = pd.to_numeric(df_p[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['True_CTR'] = (df_p['C_N'] / df_p['V_N'].replace(0, np.nan)) * 100
            df_p['True_CTR'] = df_p['True_CTR'].fillna(0.0)
            
            avg_v = df_p['V_N'].mean()
            df_p['Val_Score'] = df_p['True_CTR'] * (df_p['V_N'] / (df_p['V_N'] + avg_v))
            
            ranked_p = df_p.sort_values(by='Val_Score', ascending=False)
            ranked_p['CTR%'] = ranked_p['True_CTR'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(ranked_p[[content_col, 'CTR%', v_col, c_col, 'Val_Score']].head(10), use_container_width=True)
            
            if st.button("🚀 Run Stream 1 Engineering", key="v22_btn_p"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = ranked_p.head(10)[[content_col, 'CTR%']].to_string(index=False)
                prompt_p = f"TASK: 10 variations (7 Evo, 3 Revo) using validated DNA:\n{context_p}\nParams: {prod_description} | {keywords_input}"
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
    except Exception as e: st.error(f"S1 Error: {e}")

st.divider()

# --- STREAM 2: FORMAT ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format Strategy (Metric Unified: Clicks/Viewed)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v22_format_up")

if format_file:
    try:
        df_f = pd.read_csv(format_file)
        df_f.columns = df_f.columns.str.strip()
        
        f_cols_low = [c.lower() for c in df_f.columns]
        f_msg_idx = next((i for i, c in enumerate(f_cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
        # Unified: Looking specifically for Viewed/IMP columns in Stream 2
        f_view_idx = next((i for i, c in enumerate(f_cols_low) if any(x in c for x in ['viewed', 'imp', 'impression'])), None)
        f_click_idx = next((i for i, c in enumerate(f_cols_low) if any(x in c for x in ['clicked', 'clicks'])), None)

        if f_view_idx is not None and f_click_idx is not None:
            f_content_col, f_v_col, f_c_col = df_f.columns[f_msg_idx], df_f.columns[f_view_idx], df_f.columns[f_click_idx]
            
            df_f['V_N'] = pd.to_numeric(df_f[f_v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_f['C_N'] = pd.to_numeric(df_f[f_c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # MANDATED CALC: (Clicks/Viewed)*100
            df_f['True_CTR_F'] = (df_f['C_N'] / df_f['V_N'].replace(0, np.nan)) * 100
            df_f['True_CTR_F'] = df_f['True_CTR_F'].fillna(0.0)
            
            avg_v_f = df_f['V_N'].mean()
            df_f['Val_Score_F'] = df_f['True_CTR_F'] * (df_f['V_N'] / (df_f['V_N'] + avg_v_f))
            
            ranked_f = df_f.sort_values(by='Val_Score_F', ascending=False)
            ranked_f['CTR% Display'] = ranked_f['True_CTR_F'].apply(lambda x: f"{x:.2f}%")
            
            st.write("### 📑 Scale-Ranked Template Data")
            st.dataframe(ranked_f[[f_content_col, 'CTR% Display', f_v_col, f_c_col, 'Val_Score_F']], use_container_width=True)
            
            if st.button("🚀 Run Stream 2 Strategic Engineering (10 Row Suggestions)", key="v22_btn_f"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                format_context = ranked_f.head(10)[[f_content_col, 'CTR% Display']].to_string(index=False)
                prompt_f = f"""
                FORMAT & PERFORMANCE: {format_context}
                TASK: 10 Rows (7 Evo, 3 Revo). Replicate structural segmentation (BOGO/Hygiene), emoji usage, and style.
                Params: {prod_description} | Keywords: {keywords_input} | Segment: {seg_type}
                """
                response_f = model.generate_content(prompt_f)
                st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
        else:
            st.warning("⚠️ Stream 2 missing 'Viewed/IMP' or 'Clicked' columns. Showing raw data.")
            st.dataframe(df_f, use_container_width=True)
    except Exception as e: st.error(f"S2 Error: {e}")
