import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Unified System", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .formula-box { 
        background-color: #f8fafc; padding: 10px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.8em; font-weight: bold; border-left: 4px solid #3b82f6; color: #1e293b; margin-bottom: 8px;
    }
    .stream-header { background-color: #0f172a; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
    .summary-box { font-size: 0.82em; color: #1e293b; line-height: 1.5; background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; margin-bottom: 15px; }
    .logic-summary { font-size: 0.82em; color: #334155; line-height: 1.4; background: #f1f5f9; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: EXECUTIVE SUMMARY & LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # Pillar 1: System Overview
    st.header("🌐 System Overview")
    st.markdown("""
    <div class="summary-box">
    <b>What this tool does:</b><br>
    This platform transforms raw campaign data into high-ROI marketing copy by bridging <b>Financial Economics</b> and <b>Generative AI</b>.
    <br><br>
    <b>1. Performance Ranking:</b> Ingests CSVs to calculate real-world profitability (Revenue minus View Costs).
    <br><br>
    <b>2. DNA Extraction:</b> Identifies "Efficiency Winners"—messages that hit profit targets with the lowest capital spend.
    <br><br>
    <b>3. Result Logic:</b> Uses <i>Gemini 1.5 Pro</i> to reverse-engineer winner DNA into 10 new variations tailored to your specific targeting requirements.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pillar 2: Scoring Engine
    st.header("⚙️ Scoring & Ranking Logic")
    st.markdown(f"""
    <div class="logic-summary">
    <b>1. ROI Factor (Efficiency):</b><br>
    Calculates profit per Rupee spent. High-volume campaigns are penalized if they are "wasteful" to reach the same profit as smaller, leaner campaigns.
    <div class="formula-box">ROI = Net Profit / Total Cost</div>
    
    <b>2. Scale Validation:</b><br>
    A "Volume Confidence" multiplier is applied. This prevents 1-click flukes from ranking #1 while ensuring the winner has enough data to be scalable.
    <div class="formula-box">Final Score = ROI × [Vol / (Vol + (Avg_Vol × 0.1))]</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Pillar 3: Message Requirements (The "Who & Why")
    st.header("🎯 Message Requirements")
    keywords_input = st.text_input("Keywords", placeholder="e.g. BOGO, Sale", key="exec_kw")
    prod_description = st.text_area("Product Description", placeholder="General value prop...", height=80, key="exec_desc")
    intention = st.text_area("Intention (Inter Prompt)", placeholder="e.g. Conversion, Awareness", height=80, key="exec_int")
    
    with st.expander("📍 Target Details (Optional)", expanded=True):
        specific_product = st.text_input("Specific Product Name", key="exec_spec")
        segment = st.text_input("Segment", key="exec_seg")
        sub_segment = st.text_input("Sub-Segment", key="exec_sub")

    st.divider()

    # Pillar 4: Unit Economics (The "How Much")
    st.header("💰 Unit Economics")
    cost_per_view = st.number_input("Cost per Viewed (Rs)", value=0.66, format="%.2f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)

# --- 3. CORE PROCESSING ENGINE ---
def process_true_performance(df, label):
    df.columns = df.columns.str.strip()
    cols_low = [c.lower() for c in df.columns]
    
    msg_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['message', 'content', 'text'])), 0)
    view_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['viewed', 'imp', 'sent', 'vol'])), None)
    click_idx = next((i for i, c in enumerate(cols_low) if any(x in c for x in ['clicked', 'click', 'ctr_count'])), None)
    
    if view_idx is not None and click_idx is not None:
        content_col, v_col, c_col = df.columns[msg_idx], df.columns[view_idx], df.columns[click_idx]
        
        df['V_N'] = pd.to_numeric(df[v_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df['C_N'] = pd.to_numeric(df[c_col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        
        # Financial Calcs
        df['CTR%'] = (df['C_N'] / df['V_N'].replace(0, np.nan)) * 100
        df['Total_Cost'] = df['V_N'] * cost_per_view
        df['Net_Profit'] = (df['C_N'] * rev_per_click) - df['Total_Cost']
        df['ROI_Factor'] = (df['Net_Profit'] / df['Total_Cost'].replace(0, np.nan))
        
        avg_v = df['V_N'].mean()
        df['Final_Score'] = df['ROI_Factor'] * (df['V_N'] / (df['V_N'] + (avg_v * 0.1)))
        
        ranked = df.sort_values(by='Final_Score', ascending=False)
        ranked['CTR_Disp'] = ranked['CTR%'].fillna(0).apply(lambda x: f"{x:.2f}%")
        ranked['Score_Disp'] = ranked['Final_Score'].apply(lambda x: f"{x:.4f}")
        
        t1, t2 = st.tabs([f"📑 Full {label} Ranking", f"🏆 Top 10 Efficiency Winners"])
        with t1:
            st.dataframe(ranked[[content_col, 'CTR_Disp', v_col, c_col, 'Score_Disp']], use_container_width=True)
        with t2:
            st.table(ranked.head(10)[[content_col, 'CTR_Disp', v_col, 'Score_Disp']])
            
        return ranked, content_col
    return None, None

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

def get_ai_prompt(ranked_df, content_col):
    context = ranked_df.head(10)[[content_col, 'CTR_Disp']].to_string(index=False)
    prod_name = specific_product if specific_product else "the main product line"
    return f"""
    ROLE: Senior Growth Content Engineer.
    SPECIFIC PRODUCT: {prod_name}
    DESCRIPTION: {prod_description}
    GOAL: {intention}
    TARGETING: {segment} | {sub_segment}
    KEYWORDS: {keywords_input}
    
    DNA (Top 10 Performers):
    {context}
    
    TASK: Generate 10 Variations (7 Evolutionary, 3 Revolutionary).
    RULES:
    - Evolutionary: Optimize CTA while keeping DNA intact.
    - Revolutionary: Pivot the angle based on {intention}.
    - Formatting: Replicate structural segmentation and emoji styles.
    """

# --- 4. MAIN DASHBOARD ---
st.title("📊 Strategic Content Dashboard")

# Stream 1
st.markdown('<div class="stream-header">📂 STREAM 1: Performance-Led Innovation</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 CSVs", type="csv", accept_multiple_files=True, key="s1_exec")
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, c_s1 = process_true_performance(df_s1, "Stream 1")
    if ranked_s1 is not None and st.button("🚀 Run S1 Strategic Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(get_ai_prompt(ranked_s1, c_s1))
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Style & Format Replication</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format CSV", type="csv", key="s2_exec")
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, c_s2 = process_true_performance(df_s2, "Stream 2")
    if ranked_s2 is not None and st.button("🚀 Run S2 Style Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(get_ai_prompt(ranked_s2, c_s2))
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
