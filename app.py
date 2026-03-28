import streamlit as st
import pandas as pd
import google.generativeai as genai
import numpy as np
import re

# --- 1. CONFIG & UI ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Gemini 3 Flash", layout="wide")

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

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # Pillar 1: Message Requirements
    st.header("🎯 Message Requirements")
    keywords_input = st.text_input("Keywords", placeholder="e.g. BOGO, Sale", key="g3_kw")
    prod_description = st.text_area("Product Description", placeholder="General value prop...", height=80, key="g3_desc")
    intention = st.text_area("Intention (Inter Prompt)", placeholder="e.g. Conversion, Awareness", height=80, key="g3_int")
    
    with st.expander("📍 Target Details (Optional)", expanded=True):
        specific_product = st.text_input("Specific Product Name", key="g3_spec")
        segment = st.text_input("Segment", key="g3_seg")
        sub_segment = st.text_input("Sub-Segment", key="g3_sub")

    st.divider()

    # Pillar 2: Unit Economics
    st.header("💰 Unit Economics")
    cost_per_view = st.number_input("Cost per Viewed (Rs)", value=0.66, format="%.2f")
    rev_per_click = st.number_input("Revenue per Click (Rs)", value=1000.0)

    st.divider()

    # Pillar 3: System Overview
    st.header("🌐 System Overview")
    st.markdown("""
    <div class="summary-box">
    <b>Powered by Gemini 3 Flash:</b> This platform transforms raw campaign data into high-ROI copy by bridging Financial Economics and Generative AI.
    </div>
    """, unsafe_allow_html=True)

    # Pillar 4: Scoring Logic
    st.header("⚙️ Scoring & Ranking Logic")
    st.markdown(f"""
    <div class="logic-summary">
    <b>ROI Factor:</b> Net Profit / Total Cost.<br>
    <b>Final Score:</b> ROI × [Vol / (Vol + (Avg_Vol × 0.1))]
    </div>
    """, unsafe_allow_html=True)

# --- 3. PROCESSING ENGINE ---
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
s1_files = st.file_uploader("Upload S1 Performance CSVs", type="csv", accept_multiple_files=True, key="g3_s1")

# Use try-except to handle data loading gracefully
try:
    if s1_files:
        df_s1 = pd.concat([pd.read_csv(f) for f in s1_files], ignore_index=True)
        ranked_s1, c_s1 = process_data(df_s1, "Stream 1")
    else:
        ranked_s1, c_s1 = None, None
except Exception as e:
    st.error(f"Error loading Stream 1: {e}")
    ranked_s1, c_s1 = None, None

st.divider()

# Stream 2
st.markdown('<div class="stream-header">📂 STREAM 2: Structural Style DNA</div>', unsafe_allow_html=True)
s2_file = st.file_uploader("Upload S2 Style CSV", type="csv", key="g3_s2")

try:
    if s2_file:
        df_s2 = pd.read_csv(s2_file)
        ranked_s2, c_s2 = process_data(df_s2, "Stream 2")
    else:
        ranked_s2, c_s2 = None, None
except Exception as e:
    st.error(f"Error loading Stream 2: {e}")
    ranked_s2, c_s2 = None, None

st.divider()

# --- THE MASTER GENERATE BUTTON ---
if st.button("🚀 MASTER GENERATE: SYNTHESIZE PERFORMANCE & STYLE"):
    if not (ranked_s1 is not None or ranked_s2 is not None):
        st.error("Please upload valid data to generate content.")
    else:
        try:
            genai.configure(api_key=ACTIVE_KEY)
            # UPDATED FOR GEMINI 3 FLASH
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            s1_ctx = ranked_s1.head(10)[[c_s1, 'CTR_Disp']].to_string(index=False) if ranked_s1 is not None else "N/A"
            s2_ctx = ranked_s2.head(5)[[c_s2]].to_string(index=False) if ranked_s2 is not None else "N/A"
            
            master_prompt = f"""
            ROLE: Senior Growth Content Engineer.
            SPECIFIC PRODUCT: {specific_product if specific_product else 'Main Line'}
            DESCRIPTION: {prod_description}
            GOAL: {intention}
            TARGETING: {segment} | {sub_segment}
            KEYWORDS: {keywords_input}

            DNA S1 (ROI Performance): {s1_ctx}
            DNA S2 (Style/Format): {s2_ctx}

            TASK: Generate 10 variations (7 Evolutionary, 3 Revolutionary). 
            Merge S1 high-ROI angles with S2 structure and emoji style.
            """
            
            with st.spinner("Synthesizing Strategy with Gemini 3 Flash..."):
                res = model.generate_content(master_prompt)
                st.markdown("### 🏆 Master Engineered Content")
                st.markdown(highlight_keywords(res.text, keywords_input), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
