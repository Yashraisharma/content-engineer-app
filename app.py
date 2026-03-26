import streamlit as st
import pandas as pd
from google import genai
import numpy as np

# --- 1. CONFIG ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
st.set_page_config(page_title="Sure Shot Attribution", layout="wide")

# Custom CSS for Logic Transparency
st.markdown("""
    <style>
    .logic-box { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; margin-bottom: 20px; }
    .formula-text { font-family: 'Courier New', monospace; color: #007bff; font-weight: bold; font-size: 1.1em; }
    .stButton>button { background-color: #28a745; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS & OPTIONAL TARGETING ---
with st.sidebar:
    st.title("🛡️ Logic-First Engineer")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="Refill, Safety, Insulin")
    prod_description = st.text_area("Product Details", height=100)
    intention = st.text_area("Primary Goal", height=100)

    st.divider()
    
    st.header("🔍 Advanced Targeting (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed")
    seg_reason = st.text_input("2. Reason", placeholder="e.g. 30-Day Gap")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Insulin")
    spec_prod = st.text_input("4. Specific Product", placeholder="e.g. Lantus")

# --- 3. MAIN DASHBOARD ---
st.title("📊 Relational Campaign Attribution")

# LOGIC EXPLANATION (Visible to User)
st.markdown("### 🧠 The Ranking Logic")
st.markdown("""
<div class="logic-box">
    <b>Definitive Formula:</b><br>
    <span class="formula-text">Final Score = (CTR × 0.6) + (Log_Scale × 0.2) + (Relevance_Boost × 0.2)</span><br><br>
    <ul>
        <li><b>CTR (Efficiency):</b> Measures the raw click-power of the message.</li>
        <li><b>Log_Scale (Confidence):</b> Rewards messages that maintained high performance over larger audience volumes.</li>
        <li><b>Relevance (Segment Match):</b> Cross-references your 'Who Query' with the 'Advanced Targeting' boxes.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Full-Header Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()

        col_map = {c.lower(): c for c in df.columns}
        id_col = col_map.get('campaign id', None)
        msg_col = col_map.get('message', None)
        who_col = col_map.get('who query', None)
        sent_col = col_map.get('total sent(users)', None)
        clicks_col = col_map.get('total clicked(users)', None)

        if all([id_col, msg_col, sent_col, clicks_col]):
            # 1. CLEANING & METRIC CALCULATION
            df['Sent_N'] = pd.to_numeric(df[sent_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # 2. CALCULATE LOGIC COMPONENTS
            # CTR component
            df['CTR_Calc'] = (df['Clicks_N'] / df['Sent_N']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # Scale component (Logarithmic normalization)
            max_sent = df['Sent_N'].max() if df['Sent_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Sent_N']) / np.log1p(max_sent)
            
            # Relevance component (Fuzzy match with optional boxes)
            target_keywords = f"{seg_type} {seg_reason} {sub_seg} {spec_prod}".lower().split()
            if who_col and target_keywords:
                df['Rel_Match'] = df[who_col].apply(lambda x: 1.0 if any(k in str(x).lower() for k in target_keywords) else 0.0)
            else:
                df['Rel_Match'] = 0.5 # Neutral if no targeting provided

            # 3. FINAL DEFINITIVE RANKING
            df['Final_Ranking_Score'] = (df['CTR_Calc'] * 0.6) + (df['Log_Scale'] * 0.2) + (df['Rel_Match'] * 0.2)

            # --- DISPLAY ---
            st.markdown("### 🏆 Top 10 Ranked Benchmark Campaigns")
            winners = df.sort_values(by='Final_Ranking_Score', ascending=False).head(10).copy()
            
            st.table(winners[[id_col, who_col, msg_col, 'CTR_Calc', 'Final_Ranking_Score']])

            if st.button("🚀 Engineer Content from these Benchmarks"):
                client = genai.Client(api_key=ACTIVE_KEY)
                context = ""
                for _, row in winners.iterrows():
                    context += f"ID: {row[id_col]} | Segment: {row.get(who_col)} | CTR: {row['CTR_Calc']:.4%} | Message: {row[msg_col]}\n"

                prompt = f"""
                Analyze the Success DNA of these Campaign IDs for the segment: {seg_type} {sub_seg}.
                Goal: {intention}
                Context: {context}
                
                Generate 10 variations (7 Evolutionary, 3 Revolutionary) that maintain the high-confidence logic of the benchmarks.
                Output a table with Campaign ID reference and Reactivity Reasoning.
                """
                response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                st.markdown(response.text)
        else:
            st.error("CSV must contain: Campaign ID, Message, Total Sent(users), and Total Clicked(users).")

    except Exception as e:
        st.error(f"Logic Error: {e}")
