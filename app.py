import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Metric Precision", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    section[data-testid="stSidebar"] { width: 400px !important; }
    .formula-box { 
        background-color: #f0f2f6; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.95em; font-weight: bold; border-left: 6px solid #28a745; color: #1f2937; margin-bottom: 10px;
    }
    .stream-header {
        background-color: #1f2937; color: white; padding: 15px; border-radius: 5px; margin-top: 25px; font-weight: bold; font-size: 1.2em;
    }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale", key="v15_kw")
    prod_description = st.text_area("Product Details", height=100, key="v15_prod")
    intention = st.text_area("Primary Goal", height=100, key="v15_goal")
    
    st.divider()
    st.header("🔍 Advanced Segmentation")
    seg_type = st.text_input("Segment Type", key="v15_type")
    sub_seg = st.text_input("Sub Segment", key="v15_sub")
    spec_prod = st.text_input("Specific Product Base", key="v15_base")

    st.divider()
    st.header("📋 Project Summary")
    st.write("Extracts proven success DNA from raw CSV metrics to engineer 10 optimized content variations.")
    
    st.header("⚙️ Applied Logic")
    with st.expander("1. Firm Ranking Engine", expanded=True):
        st.markdown('<div class="formula-box">CTR% = (Clicked / Viewed) × 100</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">Power Score = (CTR% × 0.7) + (Vol × 0.3)</div>', unsafe_allow_html=True)

    with st.expander("2. 7+3 Strategic Engineering", expanded=True):
        st.write("**7 Evolutionary:** Based on Format Skeletons.")
        st.write("**3 Revolutionary:** New psychological hooks.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Content Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- STREAM 1: PERFORMANCE DNA ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance-Based Analysis (Historical Winners)</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Historical Performance CSVs", type="csv", accept_multiple_files=True, key="v15_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        # Mapping for specific column names
        cols_low = [c.lower() for c in df_p.columns]
        msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
        view_idx = next((i for i, c in enumerate(cols_low) if 'viewed(users)' in c or 'viewed' in c), None)
        click_idx = next((i for i, c in enumerate(cols_low) if 'clicked(users)' in c or 'clicked' in c), None)

        if view_idx is not None and click_idx is not None:
            content_col = df_p.columns[msg_idx]
            v_col, c_col = df_p.columns[view_idx], df_p.columns[click_idx]
            
            # --- MANDATED CALCULATION: (Clicks/Viewed)*100 ---
            df_p['V_N'] = pd.to_numeric(df_p[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['C_N'] = pd.to_numeric(df_p[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['CTR_Final'] = (df_p['C_N'] / df_p['V_N'].replace(0, np.nan)) * 100
            df_p['CTR_Final'] = df_p['CTR_Final'].fillna(0.0)
            
            # Power Score for Ranking
            df_p['Power_Score'] = (df_p['CTR_Final'] * 0.7) + ((df_p['V_N'] / (df_p['V_N'].max() or 1)) * 30)
            
            full_ranked = df_p.sort_values(by='Power_Score', ascending=False)
            full_ranked['CTR% Display'] = full_ranked['CTR_Final'].apply(lambda x: f"{x:.2f}%")
            
            t1, t2 = st.tabs(["📑 All Rows (Metric-Ranked)", "🏆 Top 10 Winners"])
            with t1: st.dataframe(full_ranked[[content_col, 'CTR% Display', v_col, c_col, 'Power_Score']], use_container_width=True)
            with t2: st.table(full_ranked.head(10)[[content_col, 'CTR% Display', v_col, 'Power_Score']])
            
            if st.button("🚀 Run Stream 1 Strategic Engineering", key="v15_btn_p"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = full_ranked.head(10)[[content_col, 'CTR% Display']].to_string(index=False)
                prompt_p = f"TASK: Generate 10 variations (7 Evo, 3 Revo) using this Performance DNA:\n{context_p}\nParams: Product: {prod_description} | Keywords: {keywords_input} | Segment: {seg_type}"
                res_p = model.generate_content(prompt_p)
                st.markdown(highlight_keywords(res_p.text, keywords_input), unsafe_allow_html=True)
        else:
            st.error(f"Metric Mapping Failed. Ensure columns 'Total Clicked(users)' and 'Total Viewed(users)' exist.")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# --- STREAM 2: FORMAT STRATEGY ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format-Based Engineering (Strategic Style)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v15_format_up")

if format_file:
    df_f = pd.read_csv(format_file)
    st.write("### 📝 Full Format Dataset")
    st.dataframe(df_f, use_container_width=True)
    
    if st.button("🚀 Run Stream 2 Strategic Engineering (10 Row Suggestions)", key="v15_btn_f"):
        if not ACTIVE_KEY:
            st.error("API Key Missing")
        else:
            with st.spinner("Engineering Style-Specific Variations..."):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                format_context = df_f.to_string(index=False)
                
                prompt_f = f"""
                REFERENCE FORMAT DATA: {format_context}
                TASK: Generate 10 Content Rows. 7 Evolutionary, 3 Revolutionary.
                Parameters: 
                Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                Segment: {seg_type} | {sub_seg} | {spec_prod}
                
                STRICT INSTRUCTION: Replicate the emoji usage, structural segmentation, and character style.
                OUTPUT: Markdown table with: Usage Rank, New Content (File 2 Style), Segmentation Validation, Reference ID, Hit Percentage, Logic Hook.
                """
                
                response_f = model.generate_content(prompt_f)
                st.success("✅ Engineering Complete")
                st.markdown(highlight_keywords(response_f.text, keywords_input), unsafe_allow_html=True)
