import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Tabbed Suite", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .formula-box { 
        background-color: #f0f2f6; padding: 15px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.95em; font-weight: bold; border-left: 6px solid #10b981; color: #1f2937; margin-bottom: 10px;
    }
    .stream-header {
        background-color: #1f2937; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; font-size: 1.1em;
    }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    st.header("🎯 Parameters")
    keywords_input = st.text_input("Keywords", key="tab_kw")
    prod_description = st.text_area("Product Details", height=100, key="tab_prod")
    seg_type = st.text_input("Segment Type", key="tab_seg")
    
    st.divider()
    st.header("⚙️ Scaling Logic")
    with st.expander("Confidence Engine", expanded=True):
        st.markdown('<div class="formula-box">Score = CTR% * (Vol / (Vol + Avg_Vol))</div>', unsafe_allow_html=True)
        st.caption("Prioritizes high-scale proven winners (e.g. 15k views / 43% CTR).")

# --- 3. PROCESSING LOGIC ---
def process_and_rank(df, stream_label):
    df.columns = df.columns.str.strip()
    cols_low = [c.lower() for c in df.columns]
    
    # Mapping
    msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text', 'body'])), 0)
    view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'impression', 'sent', 'delivered', 'volume'])), None)
    click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'click', 'ctr_count'])), None)
    
    if view_idx is not None and click_idx is not None:
        content_col, v_col, c_col = df.columns[msg_idx], df.columns[view_idx], df.columns[click_idx]
        
        # Clean & Calc
        df['V_N'] = pd.to_numeric(df[v_col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce').fillna(0)
        df['C_N'] = pd.to_numeric(df[c_col].astype(str).str.replace(',', '').str.replace('%', ''), errors='coerce').fillna(0)
        df['True_CTR'] = (df['C_N'] / df['V_N'].replace(0, np.nan)) * 100
        df['True_CTR'] = df['True_CTR'].fillna(0.0)
        
        avg_v = df['V_N'].mean()
        df['Confidence_Score'] = df['True_CTR'] * (df['V_N'] / (df['V_N'] + avg_v))
        
        ranked = df.sort_values(by='Confidence_Score', ascending=False)
        ranked['CTR% Display'] = ranked['True_CTR'].apply(lambda x: f"{x:.2f}%")
        
        # TABBED UI
        t1, t2 = st.tabs([f"📑 Full {stream_label} Ranked", f"🏆 Top 10 {stream_label} Winners"])
        with t1:
            st.dataframe(ranked[[content_col, 'CTR% Display', v_col, c_col, 'Confidence_Score']], use_container_width=True)
        with t2:
            st.table(ranked.head(10)[[content_col, 'CTR% Display', v_col, 'Confidence_Score']])
            
        return ranked, content_col
    else:
        st.error(f"❌ Mapping Failed for {stream_label}. Found: {list(df.columns)}")
        return None, None

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- 4. MAIN DASHBOARD ---
st.title("📊 Strategic Scale-Validated Dashboard")

# Stream 1
st.markdown('<div class="stream-header">📂 STREAM 1: Historical Performance Analysis</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 CSVs", type="csv", accept_multiple_files=True, key="tab_s1_up")
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, content_s1 = process_and_rank(df_s1, "Stream 1")
    if ranked_s1 is not None and st.button("🚀 Run Stream 1 Engineering", key="btn_s1"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s1.head(10)[[content_s1, 'CTR% Display']].to_string(index=False)
        res = model.generate_content(f"TASK: 10 variations (7 Evo, 3 Revo). DNA: {context}. Params: {prod_description} | {keywords_input}")
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Format & Style Strategy</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format CSV", type="csv", key="tab_s2_up")
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, content_s2 = process_and_rank(df_s2, "Stream 2")
    if ranked_s2 is not None and st.button("🚀 Run Stream 2 Strategic Engineering", key="btn_s2"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s2.head(10)[[content_s2, 'CTR% Display']].to_string(index=False)
        res = model.generate_content(f"FORMAT: {context}\nTASK: 10 Rows (7 Evo, 3 Revo). Style clone. Params: {prod_description} | {keywords_input}")
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
