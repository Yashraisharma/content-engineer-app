import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Strategic Flow", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3.5em; background-color: #007bff; color: white; font-weight: bold; }
    .formula-box { 
        background-color: #f8fafc; padding: 10px; border-radius: 8px; font-family: 'Courier New', monospace; 
        font-size: 0.8em; font-weight: bold; border-left: 4px solid #3b82f6; color: #1e293b; margin-bottom: 8px;
    }
    .stream-header { background-color: #0f172a; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
    .logic-summary { font-size: 0.82em; color: #334155; line-height: 1.4; background: #f1f5f9; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    .prompt-history { font-size: 0.75em; color: #64748b; font-style: italic; background: #ffffff; padding: 8px; border-radius: 5px; border-left: 3px solid #cbd5e1; margin-bottom: 5px; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: REORGANIZED FLOW ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # --- SECTION 1: MESSAGE REQUIREMENTS (TOP) ---
    st.header("🎯 Message Requirements")
    keywords_input = st.text_input("Keywords", placeholder="e.g. BOGO, Sale", key="flow_kw")
    
    # Symmetrical Text Areas
    prod_description = st.text_area("Product Description", placeholder="General product value prop...", height=80, key="flow_desc")
    intention = st.text_area("Intention (Inter Prompt)", placeholder="e.g. Conversion, Awareness, Upsell", height=80, key="flow_int")
    
    # Optional Targeting Details
    with st.expander("📍 Target Details (Optional)", expanded=True):
        specific_product = st.text_input("Specific Product Name", placeholder="e.g. iPhone 15 Pro", key="flow_spec")
        segment = st.text_input("Segment", placeholder="e.g. Existing Users", key="flow_seg")
        sub_segment = st.text_input("Sub-Segment", placeholder="e.g. High-Spenders", key="flow_sub")

    st.divider()

    # --- SECTION 2: UNIT ECONOMICS (BELOW REQUIREMENTS) ---
    st.header("💰 Unit Economics")
    cost_per_view = st.number_input("Cost per Viewed (Rs)", value=0.66, format="%.2f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)
    
    st.divider()
    
    # --- PERFORMANCE SCORING SUMMARY ---
    st.header("⚙️ Scoring Logic")
    st.markdown(f"""
    <div class="logic-summary">
    <b>ROI Factor:</b> Efficiency of profit vs cost.<br>
    <b>Final Score:</b> ROI × Volume Confidence Factor.
    </div>
    """, unsafe_allow_html=True)

    # --- STRATEGIC PROMPT HISTORY ---
    st.header("📜 Strategic Prompt History")
    history = [
        "1. Fix Column Mapping (Viewed/Clicked).",
        "2. UI Tabs for Full Ranking vs Top 10.",
        "3. Financials: 1000 Rev/Click | 0.66 Cost/View.",
        "4. Efficiency Logic: Lower volume wins if profits match.",
        "5. Final Score visibility per row.",
        "6. Symmetrical UI: Intention and Product Description boxes equalized.",
        "7. Layout: Unit Economics moved below Message Requirements."
    ]
    for p in history:
        st.markdown(f'<div class="prompt-history">{p}</div>', unsafe_allow_html=True)

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
st.markdown('<div class="stream-header">📂 STREAM 1: Performance Analysis</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload S1 CSVs", type="csv", accept_multiple_files=True, key="s1_flow")
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, c_s1 = process_true_performance(df_s1, "Stream 1")
    if ranked_s1 is not None and st.button("🚀 Run S1 Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(get_ai_prompt(ranked_s1, c_s1))
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Style & Format Replication</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Format CSV", type="csv", key="s2_flow")
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, c_s2 = process_true_performance(df_s2, "Stream 2")
    if ranked_s2 is not None and st.button("🚀 Run S2 Engineering"):
        genai.configure(api_key=ACTIVE_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(get_ai_prompt(ranked_s2, c_s2))
        st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
