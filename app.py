import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Avg-Baseline Edition", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 400px !important; }
    .formula-box { 
        background-color: #f0f2f6; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.95em; font-weight: bold; border-left: 6px solid #6366f1; color: #1f2937; margin-bottom: 10px;
    }
    .stream-header {
        background-color: #1f2937; color: white; padding: 15px; border-radius: 5px; margin-top: 25px; font-weight: bold; font-size: 1.2em;
    }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & AVG-LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale", key="v18_kw")
    prod_description = st.text_area("Product Details", height=100, key="v18_prod")
    intention = st.text_area("Primary Goal", height=100, key="v18_goal")
    
    st.divider()
    st.header("🔍 Segmentation")
    seg_type = st.text_input("Segment Type", key="v18_type")
    sub_seg = st.text_input("Sub Segment", key="v18_sub")

    st.divider()
    st.header("⚙️ Efficiency Logic (v2)")
    with st.expander("Avg-Volume Baseline", expanded=True):
        st.markdown('<div class="formula-box">Baseline = Dataset Avg Volume</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">Efficiency = CTR% / (Vol / Avg_Vol)</div>', unsafe_allow_html=True)
        st.caption("Highlights content that beats the average efficiency of the current dataset.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Efficiency Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE DNA ---
st.markdown('<div class="stream-header">📂 STREAM 1: Average-Baseline Efficiency Analysis</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Performance CSVs", type="csv", accept_multiple_files=True, key="v18_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        cols_low = [c.lower() for c in df_p.columns]
        msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
        view_idx = next((i for i, c in enumerate(cols_low) if 'viewed(users)' in c or 'viewed' in c), None)
        click_idx = next((i for i, c in enumerate(cols_low) if 'clicked(users)' in c or 'clicked' in c), None)

        if view_idx is not None and click_idx is not None:
            content_col = df_p.columns[msg_idx]
            v_col, c_col = df_p.columns[view_idx], df_p.columns[click_idx]
            
            # --- CLEANING ---
            df_p['V_N'] = pd.to_numeric(df_p[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['C_N'] = pd.to_numeric(df_p[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # --- MATH ---
            df_p['True_CTR'] = (df_p['C_N'] / df_p['V_N'].replace(0, np.nan)) * 100
            df_p['True_CTR'] = df_p['True_CTR'].fillna(0.0)
            
            # NEW LOGIC: Use Dataset Average Volume
            avg_vol = df_p['V_N'].mean() if df_p['V_N'].mean() > 0 else 1
            
            # Efficiency Rank: CTR relative to how much volume it 'consumed' vs the average
            df_p['Efficiency_Index'] = df_p['True_CTR'] / (df_p['V_N'] / avg_vol)
            
            full_ranked = df_p.sort_values(by='Efficiency_Index', ascending=False)
            full_ranked['CTR% Display'] = full_ranked['True_CTR'].apply(lambda x: f"{x:.2f}%")
            
            t1, t2 = st.tabs(["📑 All Rows (Efficiency Ranked)", "🏆 Top 10 High-Efficiency Winners"])
            with t1: st.dataframe(full_ranked[[content_col, 'CTR% Display', v_col, c_col, 'Efficiency_Index']], use_container_width=True)
            with t2: st.table(full_ranked.head(10)[[content_col, 'CTR% Display', v_col, 'Efficiency_Index']])
            
            if st.button("🚀 Run Stream 1 Efficiency Engineering", key="v18_btn_p"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = full_ranked.head(10)[[content_col, 'CTR% Display']].to_string(index=False)
                prompt_p = f"Using Avg-Baseline Efficiency DNA:\n{context_p}\nEngineer 10 variations (7 Evo, 3 Revo) for {prod_description} with keywords {keywords_input}."
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
        else:
            st.error(f"Metric Mapping Failed. Found: {list(df_p.columns)}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# --- STREAM 2: FORMAT STRATEGY ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format-Based Engineering (Strategic Style)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v18_format_up")

if format_file:
    df_f = pd.read_csv(format_file)
    st.write("### 📝 Full Format Dataset")
    st.dataframe(df_f, use_container_width=True)
    
    if st.button("🚀 Run Stream 2 Strategic Engineering (10 Row Suggestions)", key="v18_btn_f"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        format_context = df_f.to_string(index=False)
        prompt_f = f"""
        FORMAT: {format_context}
        TASK: 10 Content Rows. 7 Evolutionary, 3 Revolutionary.
        PARAMS: {prod_description} | Keywords: {keywords_input} | Goal: {intention} | Segment: {seg_type}
        STRICT: Replicate structural segmentation (e.g. hygiene/BOGO format), emoji usage, and character style.
        """
        response_f = model.generate_content(prompt_f)
        st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
