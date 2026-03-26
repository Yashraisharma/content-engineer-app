import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Financial Suite", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .formula-box { 
        background-color: #f0f2f6; padding: 12px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.85em; font-weight: bold; border-left: 5px solid #10b981; color: #1f2937; margin-bottom: 8px;
    }
    .stream-header { background-color: #1f2937; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
    .sidebar-desc { font-size: 0.9em; color: #4b5563; line-height: 1.4; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & METHODOLOGY EXPLAINER ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("💰 Unit Economics")
    cost_per_vol = st.number_input("Cost per Volume (Rs)", value=0.6, format="%.4f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Keywords", key="final_kw")
    prod_description = st.text_area("Product Details", height=80, key="final_prod")
    seg_type = st.text_input("Segment Type", key="final_seg")

    st.divider()
    
    # --- METHODOLOGY EXPLAINER (Bottom of Sidebar) ---
    st.header("📖 Methodology & UI Guide")
    
    with st.expander("🚀 The 'Better Message' Logic", expanded=True):
        st.markdown('<div class="sidebar-desc">We no longer rank by CTR alone. A message is "Better" if it maintains profit at <b>Scale</b>.</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">Net Profit = (Clicks × Rev) - (Vol × Cost)</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">BVS = Net Profit × [Vol / (Vol + Avg_Vol)]</div>', unsafe_allow_html=True)
        st.caption("BVS (Business Value Score) punishes 'lucky' low-volume flukes and rewards 'proven' high-volume winners.")

    with st.expander("📊 Scale-Confidence Rule", expanded=False):
        st.write("If **Campaign A (1.4M views)** and **Campaign B (400k views)** have equal profit, A wins.")
        st.write("Why? Because A has been 'stress-tested' against a larger audience, making it a lower-risk bet for future scaling.")

    with st.expander("📑 UI Navigation Help", expanded=False):
        st.write("**Full Business Rank Tab:** Every row from your CSV, ranked by BVS.")
        st.write("**Top 10 Winners Tab:** A clean snapshot of the best performers used for AI generation.")
        st.write("**Profit_Disp:** The raw Rupee value earned/lost by that specific message.")

# --- 3. CORE PROCESSING ENGINE ---
def process_bvs_rank(df, label):
    df.columns = df.columns.str.strip()
    cols_low = [c.lower() for c in df.columns]
    
    msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
    view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'sent', 'vol'])), None)
    click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'click'])), None)
    
    if view_idx is not None and click_idx is not None:
        content_col, v_col, c_col = df.columns[msg_idx], df.columns[view_idx], df.columns[click_idx]
        
        df['V_N'] = pd.to_numeric(df[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['C_N'] = pd.to_numeric(df[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        df['CTR%'] = (df['C_N'] / df['V_N'].replace(0, np.nan)) * 100
        df['Net_Profit'] = (df['C_N'] * rev_per_click) - (df['V_N'] * cost_per_vol)
        
        avg_v = df['V_N'].mean()
        df['BVS'] = df['Net_Profit'] * (df['V_N'] / (df['V_N'] + avg_v))
        
        ranked = df.sort_values(by='BVS', ascending=False)
        ranked['CTR_Disp'] = ranked['CTR%'].fillna(0).apply(lambda x: f"{x:.2f}%")
        ranked['Profit_Disp'] = ranked['Net_Profit'].apply(lambda x: f"₹{x:,.0f}")
        
        # UI TABS IMPLEMENTATION
        t1, t2 = st.tabs([f"📑 Full {label} Business Rank", f"🏆 Top 10 High-Value Winners"])
        with t1:
            st.dataframe(ranked[[content_col, 'CTR_Disp', v_col, 'Profit_Disp', 'BVS']], use_container_width=True)
        with t2:
            st.table(ranked.head(10)[[content_col, 'CTR_Disp', v_col, 'Profit_Disp', 'BVS']])
            
        return ranked, content_col
    return None, None

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- 4. MAIN DASHBOARD ---
st.title("📊 Strategic Financial Dashboard")

# Stream 1
st.markdown('<div class="stream-header">📂 STREAM 1: Historical Performance Analysis</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 Performance CSVs", type="csv", accept_multiple_files=True, key="s1_final")
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, c_s1 = process_bvs_rank(df_s1, "Stream 1")
    if ranked_s1 is not None and st.button("🚀 Run S1 Strategic Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s1.head(10)[[c_s1, 'CTR_Disp']].to_string(index=False)
        prompt = f"TASK: 10 variations (7 Evo, 3 Revo) using Profit DNA:\n{context}\nParams: {prod_description} | {keywords_input} | {seg_type}"
        res = model.generate_content(prompt)
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Format Strategy & Style Replication</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format CSV", type="csv", key="s2_final")
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, c_s2 = process_bvs_rank(df_s2, "Stream 2")
    if ranked_s2 is not None and st.button("🚀 Run S2 Strategic Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s2.head(10)[[c_s2, 'CTR_Disp']].to_string(index=False)
        prompt = f"""
        FORMAT & DNA: {context}
        TASK: 10 Rows (7 Evo, 3 Revo). Replicate structural segmentation (BOGO/Hygiene) and emojis.
        Params: {prod_description} | {keywords_input} | {seg_type}
        """
        res = model.generate_content(prompt)
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
