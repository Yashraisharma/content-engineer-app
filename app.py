import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | G3 Flash Edition", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 4em; background-color: #059669; color: white; font-weight: bold; font-size: 1.1em; border: none; margin-top: 20px; }
    .stButton>button:hover { background-color: #047857; border: none; }
    .stream-header { background-color: #0f172a; color: white; padding: 12px; border-radius: 5px; margin-top: 20px; font-weight: bold; }
    .summary-box { font-size: 0.82em; color: #1e293b; line-height: 1.5; background: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #dee2e6; margin-bottom: 15px; }
    .logic-summary { font-size: 0.82em; color: #334155; line-height: 1.4; background: #f1f5f9; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 10px; }
    .formula-box { background-color: #f8fafc; padding: 8px; border-radius: 5px; font-family: monospace; font-size: 0.85em; border-left: 3px solid #3b82f6; margin-top: 5px; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: STRATEGIC INPUTS ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # Pillar 1: Message Requirements
    st.header("🎯 Message Requirements")
    keywords_input = st.text_input("Keywords", placeholder="e.g. BOGO, Sale", key="f_kw")
    prod_description = st.text_area("Product Description", placeholder="Copy-paste the specific offer/product details here...", height=100, key="f_desc")
    intention = st.text_area("Intention (Inter Prompt)", placeholder="e.g. Conversion, Reactivation", height=80, key="f_int")
    
    # Pillar 2: Target Details (Segment Intelligence)
    with st.expander("📍 Target Details (Segment Intelligence)", expanded=True):
        specific_product = st.text_input("Specific Product Name")
        st.divider()
        segment = st.text_input("Segment Name")
        seg_desc = st.text_area("Segment Description", placeholder="e.g. Chronic patients, high-value...", height=70)
        st.divider()
        sub_segment = st.text_input("Sub-Segment Name")
        sub_desc = st.text_area("Sub-Segment Description", placeholder="e.g. Price sensitive, forgetful...", height=70)
        
        circle_subscriber = st.checkbox("CIRCLE Subscriber (Tick if yes)", value=False)

    st.divider()

    # Pillar 3: Unit Economics
    st.header("💰 Unit Economics")
    cost_per_view = st.number_input("Cost per Viewed (Rs)", value=0.66, format="%.2f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)

    st.divider()

    # Pillar 4: System Overview (Bottom)
    st.header("🌐 System Overview")
    st.markdown("""
    <div class="summary-box">
    <b>What this tool does:</b><br>
    Synthesizes Historical ROI (Stream 1) and Visual Formats (Stream 2) into 10 engineered variations using <b>Gemini 3 Flash</b>.
    </div>
    """, unsafe_allow_html=True)

    # Pillar 5: Scoring Logic (Bottom)
    st.header("⚙️ Scoring & Ranking Logic")
    st.markdown(f"""
    <div class="logic-summary">
    <b>ROI Factor:</b> Net Profit / Total Cost.<br>
    <b>Final Score:</b> ROI × [Vol / (Vol + (Avg_Vol × 0.1))]
    </div>
    """, unsafe_allow_html=True)

# --- 3. CORE PROCESSING ENGINE ---
def process_data(df, label):
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
        
        st.markdown(f"### 📑 {label} Analysis")
        t1, t2 = st.tabs(["Full Ranking", "Top Efficiency Winners"])
        with t1: st.dataframe(ranked[[content_col, 'CTR_Disp', v_col, c_col, 'Score_Disp']], use_container_width=True)
        with t2: st.table(ranked.head(10)[[content_col, 'CTR_Disp', v_col, 'Score_Disp']])
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
st.title("📊 Strategic Content Dashboard")

# Stream 1
st.markdown('<div class="stream-header">📂 STREAM 1: Performance ROI DNA</div>', unsafe_allow_html=True)
s1_files = st.file_uploader("Upload Performance CSVs", type="csv", accept_multiple_files=True, key="main_s1")
ranked_s1, c_s1 = None, None
if s1_files:
    df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
    ranked_s1, c_s1 = process_data(df_s1, "Stream 1")

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Structural Style DNA</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload Style CSV", type="csv", key="main_s2")
ranked_s2, c_s2 = None, None
if s2_file:
    df_s2 = pd.read_csv(s2_file)
    ranked_s2, c_s2 = process_data(df_s2, "Stream 2")

st.divider()

# --- THE MASTER GENERATE ENGINE ---
if st.button("🚀 MASTER GENERATE: SYNTHESIZE PERFORMANCE & STYLE"):
    if not (ranked_s1 is not None or ranked_s2 is not None):
        st.error("Please upload data files before generating.")
    else:
        try:
            # Setup Model
            genai.configure(api_key=ACTIVE_KEY)
            MODEL_NAME = 'gemini-3-flash-preview' # Targeting Gemini 3 Flash
            model = genai.GenerativeModel(MODEL_NAME)
            
            # Constraints: CIRCLE & Data Integrity
            circle_logic = "STRICT: Do NOT mention 'CIRCLE' or 'Free Delivery' (Circle-only benefit)." if not circle_subscriber else "Target is a CIRCLE user. Highlight 'Unlimited Free Delivery'."
            
            integrity_logic = f"""
            DATA INTEGRITY RULES:
            - SOURCE OF TRUTH: Use ONLY the offers/discounts in the Description: '{prod_description}'.
            - DO NOT hallucinate discounts (e.g., 15% off) from the DNA samples if they aren't in the Description.
            - DO NOT name the segment '{segment}' or sub-segment '{sub_segment}' in the text.
            - AUDIENCE TRAITS: Address {seg_desc} and {sub_desc} implicitly.
            """

            # Gather DNA
            s1_ctx = ranked_s1.head(10)[[c_s1, 'CTR_Disp']].to_string(index=False) if ranked_s1 is not None else "N/A"
            s2_ctx = ranked_s2.head(10)[[c_s2]].to_string(index=False) if ranked_s2 is not None else "N/A"
            
            master_prompt = f"""
            PRODUCT: {specific_product if specific_product else 'Main Line'}
            DESCRIPTION: {prod_description}
            GOAL: {intention}
            KEYWORDS: {keywords_input}
            
            {circle_logic}
            {integrity_logic}

            PERFORMANCE DNA (Stream 1): {s1_ctx}
            STYLE DNA (Stream 2): {s2_ctx}

            TASK: Generate 10 Variations in a Markdown Table with 6 columns:
            1. New Message: The engineered copy (Replicate Stream 2 emoji/line-break style).
            2. Reference: The DNA row from Stream 1 used for the angle.
            3. Selection Logic: Why this matches the {intention}.
            4. Strategic Lift: How text/emojis improve on the reference.
            5. Segmentation Alignment: How {segment}/{sub_segment} traits were woven in.
            6. Expected CTR: Realistic projection based on Stream 1.

            CRITICAL: No introductory text. No hallucinated offers. Immediate table output.
            """
            
            with st.spinner("Synthesizing Strategy with Gemini 3 Flash..."):
                res = model.generate_content(master_prompt)
                st.markdown("### 🏆 Master Engineered Content Strategy")
                st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Execution Error: {e}")
