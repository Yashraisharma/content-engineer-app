import streamlit as st
import pandas as pd
from google import genai
import re
import io

# --- 1. CONFIG & SECURITY ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Strategic Edition", layout="wide")

# Custom UI Styling - Maximized for Sidebar Visibility
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    mark { border-radius: 4px; padding: 0 2px; }
    
    /* Global Sidebar Width and Text Wrapping */
    section[data-testid="stSidebar"] { width: 380px !important; }
    [data-testid="stSidebar"] .stMarkdown { word-break: break-word; }
    
    /* Professional Formula Box */
    .formula-box { 
        background-color: #f0f2f6; 
        padding: 15px; 
        border-radius: 8px; 
        font-family: 'Courier New', monospace; 
        font-size: 0.95em;
        font-weight: bold;
        line-height: 1.5;
        border-left: 6px solid #007bff;
        color: #1f2937;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS > SUMMARY > LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # SECTION 1: CAMPAIGN PARAMETERS
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale, Hygiene")
    
    # Both boxes set to height=150 for perfect symmetry
    prod_description = st.text_area("Product Details", height=150, placeholder="Describe the product or service...")
    intention = st.text_area("Primary Goal", height=150, placeholder="e.g. Maximize CTR, drive 500 app installs, etc.")
    
    st.divider()

    # SECTION 2: PROJECT SUMMARY
    st.header("📋 Project Summary")
    st.write("""
    This tool transforms historical campaign data into optimized content. 
    It identifies proven success patterns and evolves them into 10 high-performing variations.
    """)
    
    st.divider()
    
    # SECTION 3: APPLIED LOGIC
    st.header("⚙️ Applied Logic")
    
    with st.expander("1. Firm Ranking Engine", expanded=True):
        st.write("**Statistical Weighting:**")
        # Optimized HTML box for formula visibility
        st.markdown("""<div class="formula-box">Score = (CTR × 0.7) + (Vol × 0.3)</div>""", unsafe_allow_html=True)
        st.caption("Prioritizes high-CTR creative only if it has significant Impression scale (Volume).")

    with st.expander("2. 7+3 Engineering Strategy", expanded=True):
        st.write("**Evolutionary (7 Rows):**")
        st.caption("Maps 'Winning Skeletons' and swaps in new keywords.")
        st.write("**Revolutionary (3 Rows):**")
        st.caption("Pivots to new psychological angles while keeping quality standards.")

    with st.expander("3. Validation & Mapping", expanded=True):
        st.write("**Hit % Estimation:**")
        st.caption("AI self-audit of structural alignment with Reference IDs.")
        st.write("**Semantic Highlighting:**")
        st.caption("Visual confirmation of keyword integration.")

# --- 3. MAIN PAGE: DATA & OUTPUT ---
st.title("📊 Strategic Content Engineering Dashboard")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

uploaded_files = st.file_uploader("Upload Historical Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)
        imp_col = col_map.get('impression', col_map.get('sent', col_map.get('delivered', None)))

        if ctr_col and imp_col:
            df['CTR_C'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Imp_C'] = pd.to_numeric(df[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            df['Power_Score'] = (df['CTR_C'] * 0.7) + ((df['Imp_C'] / df['Imp_C'].max()) * 0.3)
            winners = df.sort_values(by='Power_Score', ascending=False).head(10).copy()
        else:
            winners = df.head(10).copy()

        t1, t2 = st.tabs(["📂 All Uploaded Data", "🏆 Performance Benchmarks"])
        with t1: st.dataframe(df, use_container_width=True)
        with t2: st.dataframe(winners[[content_col, ctr_col, imp_col, 'Power_Score']], use_container_width=True)

        if st.button("🚀 Run 10-Row Strategic Engineering"):
            if not ACTIVE_KEY:
                st.error("API Key not found in Secrets!")
            else:
                with st.spinner("Engineering high-fidelity variations..."):
                    client = genai.Client(api_key=ACTIVE_KEY)
                    winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                    
                    winners_context = ""
                    for _, row in winners.iterrows():
                        winners_context += f"--- {row['Ref_ID']} ---\nFULL CONTENT: {row[content_col]}\nCTR: {row.get(ctr_col, 'N/A')}\n\n"

                    prompt = f"""
                    REFERENCE DATA: {winners_context}
                    TASK: Generate 10 Content Rows. 7 Evolutionary, 3 Revolutionary.
                    Parameters: Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                    OUTPUT: Markdown table with: Usage Rank, New Content, Segmentation, Reference ID, Reference Content (Full), Hit Percentage, Reasoning.
                    """
                    
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering Complete")
                    st.markdown(highlight_keywords(response.text, keywords_input), unsafe_allow_html=True)
                    st.download_button("📥 Download Analysis", response.text, file_name="engineered_content.txt")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 To begin, upload your campaign CSV files to identify performance benchmarks.")
