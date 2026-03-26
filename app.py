import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | True Performance", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .formula-box { 
        background-color: #f8fafc; padding: 12px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.85em; font-weight: bold; border-left: 5px solid #3b82f6; color: #1e293b; margin-bottom: 8px;
    }
    .stream-header { background-color: #0f172a; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
    .sidebar-desc { font-size: 0.85em; color: #64748b; line-height: 1.4; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: THE TRUTH ENGINE ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    st.header("💰 Unit Economics")
    # Updated to your specific cost: 0.66 per view
    cost_per_view = st.number_input("Cost per Viewed (Rs)", value=0.66, format="%.2f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Keywords", key="final_v_kw")
    prod_description = st.text_area("Product Details", height=80, key="final_v_prod")

    st.divider()
    
    st.header("📖 Ranking Logic: The 'True' Winner")
    with st.expander("Why this Logic?", expanded=True):
        st.markdown('<div class="sidebar-desc">We combine <b>CTR (Engagement)</b> with <b>Net Profit (Economics)</b>.</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">1. Net Profit = (Clicks × 1000) - (Views × 0.66)</div>', unsafe_allow_html=True)
        st.markdown('<div class="formula-box">2. Efficiency = Net Profit / Total Cost</div>', unsafe_allow_html=True)
        st.write("This ranks **Campaign B (400k)** higher if it made the same profit as **Campaign A (1.4M)**, because B's message was more efficient.")

# --- 3. PROCESSING ENGINE ---
def process_true_performance(df, label):
    df.columns = df.columns.str.strip()
    cols_low = [c.lower() for c in df.columns]
    
    # Mapping
    msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
    view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'sent', 'vol'])), None)
    click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'click', 'ctr_count'])), None)
    
    if view_idx is not None and click_idx is not None:
        content_col, v_col, c_col = df.columns[msg_idx], df.columns[view_idx], df.columns[click_idx]
        
        # Numeric Clean
        df['V_N'] = pd.to_numeric(df[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['C_N'] = pd.to_numeric(df[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Metrics
        df['CTR%'] = (df['C_N'] / df['V_N'].replace(0, np.nan)) * 100
        df['Total_Cost'] = df['V_N'] * cost_per_view
        df['Net_Profit'] = (df['C_N'] * rev_per_click) - df['Total_Cost']
        
        # RANKING LOGIC: Efficiency (ROI)
        # We use ROI to ensure Campaign B (Low spend, high return) wins over Campaign A (High spend, same return)
        df['ROI_Factor'] = (df['Net_Profit'] / df['Total_Cost'].replace(0, np.nan))
        
        # Final Score: ROI weighted by a small scale factor to ensure it's not a 1-click fluke
        avg_v = df['V_N'].mean()
        df['Final_Performance_Score'] = df['ROI_Factor'] * (df['V_N'] / (df['V_N'] + (avg_v * 0.1)))
        
        ranked = df.sort_values(by='Final_Performance_Score', ascending=False)
        ranked['CTR_Disp'] = ranked['CTR%'].fillna(0).apply(lambda x: f"{x:.2f}%")
        ranked['Profit_Disp'] = ranked['Net_Profit'].apply(lambda x: f"₹{x:,.0f}")
        
        t1, t2 = st.tabs(["📑 Full Ranking", "🏆 Top 10 Winners"])
        with t1:
            st.dataframe(ranked[[content_col, 'CTR_Disp', v_col, c_col, 'Profit_Disp', 'Final_Performance_Score']], use_container_width=True)
        with t2:
            st.table(ranked.head(10)[[content_col, 'CTR_Disp', v_col, 'Profit_Disp']])
            
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
st.title("📊 True Performance Content Dashboard")

# Stream 1
st.markdown('<div class="stream-header">📂 STREAM 1: Historical Winner Analysis</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 CSVs", type="csv", accept_multiple_files=True, key="s1_true")
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, c_s1 = process_true_performance(df_s1, "Stream 1")
    if ranked_s1 is not None and st.button("🚀 Run S1 Strategic Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s1.head(10)[[c_s1, 'CTR_Disp']].to_string(index=False)
        res = model.generate_content(f"TASK: 10 variations based on Profit DNA:\n{context}\nParams: {prod_description} | {keywords_input}")
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Format Strategy & Style Replication</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format CSV", type="csv", key="s2_true")
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, c_s2 = process_true_performance(df_s2, "Stream 2")
    if ranked_s2 is not None and st.button("🚀 Run S2 Strategic Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        context = ranked_s2.head(10)[[c_s2, 'CTR_Disp']].to_string(index=False)
        res = model.generate_content(f"FORMAT: {context}\nTASK: 10 Rows (7 Evo, 3 Revo). Replicate structure and emojis.")
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
