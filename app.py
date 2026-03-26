import streamlit as st
import pandas as pd
from google import genai
import numpy as np

# --- 1. CONFIG ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)
st.set_page_config(page_title="High-CTR Engineer Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #d63384; color: white; font-weight: bold; border: none; }
    section[data-testid="stSidebar"] { width: 450px !important; }
    .logic-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #d63384; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    .formula { font-family: 'Courier New', monospace; font-weight: bold; color: #d63384; font-size: 1.1em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ High-CTR Engineer")
    
    st.header("🎯 Target Parameters")
    keywords_input = st.text_input("Core Keywords", placeholder="Refill, Safety, Insulin")
    prod_description = st.text_area("Product Details", height=150)
    intention = st.text_area("Primary Goal", height=150)

    st.divider()
    
    st.header("🔍 Advanced Targeting (Optional)")
    seg_type = st.text_input("1. Segment Type", placeholder="e.g. Lapsed")
    seg_reason = st.text_input("2. Reason", placeholder="e.g. 30-Day Gap")
    sub_seg = st.text_input("3. Sub Segment", placeholder="e.g. Insulin")
    spec_prod = st.text_input("4. Specific Product", placeholder="e.g. Lantus")

    st.divider()
    
    st.header("⚙️ Ranking Logic")
    with st.expander("🔬 High-CTR Formula", expanded=True):
        st.markdown('<div class="logic-box"><b>Sure-Shot Score:</b><br><span class="formula">(CTR*0.8) + (LogScale*0.2)</span></div>', unsafe_allow_html=True)
        st.caption("Prioritizes messages with the highest engagement DNA while filtering for statistical flukes.")

# --- 3. MAIN DASHBOARD ---
st.title("📊 High-Performance Campaign Attribution")

uploaded_files = st.file_uploader("Upload Campaign CSVs", type="csv", accept_multiple_files=True)

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
            # A. DATA CLEANING & REAL CTR CALCULATION
            df['Sent_N'] = pd.to_numeric(df[sent_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            df['Clicks_N'] = pd.to_numeric(df[clicks_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            
            # Calculate Actual CTR %
            df['Real_CTR_Perc'] = (df['Clicks_N'] / df['Sent_N']).replace([np.inf, -np.inf], 0).fillna(0)
            
            # B. LOGARITHMIC SCALE (For volume confidence)
            max_sent = df['Sent_N'].max() if df['Sent_N'].max() > 0 else 1
            df['Log_Scale'] = np.log1p(df['Sent_N']) / np.log1p(max_sent)
            
            # C. FINAL RANKING (80% CTR / 20% Scale)
            df['Attribution_Score'] = (df['Real_CTR_Perc'] * 0.8) + (df['Log_Scale'] * 0.2)

            # --- DISPLAY ---
            st.subheader("🔍 Step 1: Raw Campaign Audit (All Columns)")
            st.dataframe(df, use_container_width=True)

            st.divider()

            st.subheader("🏆 Step 2: High-CTR Benchmarks (Top 10)")
            winners = df.sort_values(by='Attribution_Score', ascending=False).head(10).copy()
            
            # Create a display column for CTR as a readable percentage
            winners['CTR %'] = (winners['Real_CTR_Perc'] * 100).round(2).astype(str) + '%'
            
            st.table(winners[[id_col, who_col, msg_col, 'CTR %', 'Attribution_Score']])

            if st.button("🚀 Step 3: Engineer 10 Variations from High-CTR DNA"):
                if not ACTIVE_KEY:
                    st.error("API Key missing!")
                else:
                    try:
                        client = genai.Client(api_key=ACTIVE_KEY)
                        context = ""
                        for _, row in winners.iterrows():
                            context += f"ID: {row[id_col]} | CTR: {row['CTR %']} | Message: {row[msg_col]}\n"

                        prompt = f"""
                        Analyze these High-CTR Winners for segment: {seg_type} {sub_seg}.
                        TARGET GOAL: {intention}
                        WINNERS DATA: {context}
                        
                        TASK:
                        1. Factual Audit: Why did these specific Campaign IDs achieve such high CTRs?
                        2. Engineer 10 Variations (7 Evolutionary, 3 Revolutionary).
                        3. Every row must show: Reference ID, New Content, Source Segment, Hit %, Reasoning.
                        """
                        
                        with st.spinner("Analyzing high-performance DNA..."):
                            response = client.models.generate_content(model="gemini-3-flash", contents=prompt)
                            st.success("✅ Engineering Complete")
                            st.markdown(response.text)
                            
                    except Exception as e:
                        if "429" in str(e):
                            st.error("⚠️ Quota Reached (20/day). Please try again tomorrow.")
                        else:
                            st.error(f"Error: {e}")
        else:
            st.error("Missing columns: Campaign ID, Message, Total Sent, or Total Clicked.")

    except Exception as e:
        st.error(f"System Error: {e}")
