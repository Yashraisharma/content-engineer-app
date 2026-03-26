import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Sure-Shot CTR", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .stream-header { background-color: #1f2937; color: white; padding: 15px; border-radius: 5px; margin-top: 25px; font-weight: bold; font-size: 1.2em; }
    .formula-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #28a745; font-family: monospace; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Growth Control")
    keywords_input = st.text_input("Target Keywords", key="v11_kw")
    prod_description = st.text_area("Product Details", height=100, key="v11_prod")
    st.divider()
    st.header("🔍 Segment Specs")
    seg_type = st.text_input("Segment Type", key="v11_type")
    sub_seg = st.text_input("Sub Segment", key="v11_sub")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Strategic Growth Engineering Dashboard")

st.markdown('<div class="formula-box"><b>Sure-Shot CTR Formula:</b> (Total Clicked / Total Viewed) * 100</div>', unsafe_allow_html=True)

# --- STREAM 1: PERFORMANCE ---
st.markdown('<div class="stream-header">📂 STREAM 1: Performance (Clicks/Viewed Ratio)</div>', unsafe_allow_html=True)
perf_files = st.file_uploader("Upload Performance CSVs", type="csv", accept_multiple_files=True, key="v11_perf_up")

if perf_files:
    try:
        all_dfs = [pd.read_csv(f) for f in perf_files]
        df_p = pd.concat(all_dfs, ignore_index=True)
        df_p.columns = df_p.columns.str.strip()
        
        # Mapping to find correct columns
        cols = [c.lower() for c in df_p.columns]
        msg_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['message', 'content', 'text'])), 0)
        view_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['viewed', 'impression', 'sent'])), None)
        click_idx = next((i for i, c in enumerate(cols) if any(x in c for x in ['clicked', 'clicks'])), None)

        if view_idx is not None and click_idx is not None:
            content_col = df_p.columns[msg_idx]
            view_col = df_p.columns[view_idx]
            click_col = df_p.columns[click_idx]
            
            # --- NUMERIC CLEANING & CTR CALCULATION ---
            df_p['Viewed_N'] = pd.to_numeric(df_p[view_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df_p['Clicked_N'] = pd.to_numeric(df_p[click_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # THE CORE CTR % FORMULA
            df_p['CTR_Percentage'] = (df_p['Clicked_N'] / df_p['Viewed_N'].replace(0, np.nan)) * 100
            df_p['CTR_Percentage'] = df_p['CTR_Percentage'].fillna(0)
            
            # Power Score for Ranking (weighted)
            df_p['Power_Score'] = (df_p['CTR_Percentage'] * 0.7) + ((df_p['Viewed_N'] / df_p['Viewed_N'].max()) * 30)
            
            winners_p = df_p.sort_values(by='Power_Score', ascending=False).head(10).copy()
            
            # Formatted for Display
            winners_p['CTR %'] = winners_p['CTR_Percentage'].apply(lambda x: f"{x:.2f}%")
            
            st.write("### 🏆 Top 10 Performance Benchmarks")
            st.table(winners_p[[content_col, 'CTR %', view_col, click_col, 'Power_Score']])
            
            if st.button("🚀 Run Stream 1 Engineering"):
                genai.configure(api_key=ACTIVE_KEY)
                model = genai.GenerativeModel('gemini-1.5-flash')
                context_p = winners_p[[content_col, 'CTR %']].to_string(index=False)
                prompt_p = f"Using Performance DNA:\n{context_p}\nEngineer 10 variations with keywords {keywords_input} for segment {seg_type}. Target: {prod_description}."
                res_p = model.generate_content(prompt_p)
                st.markdown(res_p.text)
        else:
            st.error(f"❌ Could not find 'Viewed' and 'Clicked' columns. Found: {list(df_p.columns)}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

# --- STREAM 2: FORMAT ---
st.markdown('<div class="stream-header">📂 STREAM 2: Format Strategy (Style Replication)</div>', unsafe_allow_html=True)
format_file = st.file_uploader("Upload Format Template CSV", type="csv", key="v11_format_up")

if format_file:
    df_f = pd.read_csv(format_file)
    st.dataframe(df_f.head(3), use_container_width=True)
    
    if st.button("🚀 Run Stream 2 (File 2 Style)"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        format_context = df_f.to_string(index=False)
        prompt_f = f"FORMAT: {format_context}\nTASK: Generate 10 variations replicating this emoji/segment style for {prod_description} with keywords {keywords_input}."
        response_f = model.generate_content(prompt_f)
        st.markdown(response_f.text)
