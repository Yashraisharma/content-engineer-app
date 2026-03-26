import streamlit as st
import pandas as pd
from google import genai
import re
import io

# --- 1. CONFIG & SECURITY ---
# Accessing the secret key from Streamlit Cloud Secrets
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Strategic Edition", layout="wide")

# Custom UI Styling for a professional look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR: PARAMETERS > SUMMARY > LOGIC ---
with st.sidebar:
    st.title("🛡️ Content Engineer Pro")
    
    # SECTION 1: CAMPAIGN PARAMETERS (TOP)
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale, Hygiene")
    prod_description = st.text_area("Product/Offer Details", height=120, placeholder="Describe the product or service...")
    intention = st.text_input("Primary Goal", placeholder="e.g. Maximize CTR / App Installs")
    
    st.divider()

    # SECTION 2: PROJECT SUMMARY
    st.header("📋 Project Summary")
    st.write("""
    This project is a high-fidelity growth marketing tool that transforms historical 
    campaign data into optimized content. It uses a mathematical bridge to identify 
    proven success patterns and evolves them into 10 high-performing variations.
    """)
    
    st.divider()
    
    # SECTION 3: APPLIED LOGIC (BOTTOM)
    st.header("⚙️ Applied Logic")
    
    with st.expander("1. Firm Ranking Engine", expanded=True):
        st.write("**Statistical Weighting:**")
        st.latex(r"Score = (CTR \times 0.7) + (Volume \times 0.3)")
        st.caption("""
        Prevents 'fluke' winners. We prioritize high-CTR creative only if it 
        has maintained performance over a significant audience scale (Impressions).
        """)

    with st.expander("2. 7+3 Engineering Strategy", expanded=True):
        st.write("**Evolutionary (7 Rows):**")
        st.caption("Identifies the 'Winning Skeleton' (emoji use, hook type, syntax) and maps new keywords into that proven frame.")
        st.write("**Revolutionary (3 Rows):**")
        st.caption("Pivots to new psychological angles (Social Proof, Scarcity, or Benefit-led) while maintaining quality standards.")

    with st.expander("3. Validation & Mapping", expanded=True):
        st.write("**Hit % Estimation:**")
        st.caption("AI-driven self-audit comparing structural alignment with the specific Reference ID used.")
        st.write("**Semantic Highlighting:**")
        st.caption("Regex-powered visual confirmation that all target keywords were integrated correctly.")

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
        
        # Mapping Columns
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)
        imp_col = col_map.get('impression', col_map.get('sent', col_map.get('delivered', None)))

        if ctr_col and imp_col:
            # Firm Logic Calculation
            df['CTR_Clean'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Imp_Clean'] = pd.to_numeric(df[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            df['Power_Score'] = (df['CTR_Clean'] * 0.7) + ((df['Imp_Clean'] / df['Imp_Clean'].max()) * 0.3)
            winners = df.sort_values(by='Power_Score', ascending=False).head(10).copy()
        else:
            winners = df.head(10).copy()

        # Visualization Tabs
        t1, t2 = st.tabs(["📂 All Uploaded Data", "🏆 Performance Benchmarks (Ranked Winners)"])
        with t1: st.dataframe(df, use_container_width=True)
        with t2: st.dataframe(winners[[content_col, ctr_col, imp_col, 'Power_Score']], use_container_width=True)

        if st.button("🚀 Run 10-Row Strategic Engineering"):
            if not ACTIVE_KEY:
                st.error("API Key not found in Secrets! Please check your Streamlit Cloud settings.")
            else:
                with st.spinner("Engineering high-fidelity variations using Firm Logic..."):
                    client = genai.Client(api_key=ACTIVE_KEY)
                    winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                    
                    # Preparing full context for the AI
                    winners_context = ""
                    for _, row in winners.iterrows():
                        winners_context += f"--- {row['Ref_ID']} ---\nFULL CONTENT: {row[content_col]}\nCTR: {row.get(ctr_col, 'N/A')}\n\n"

                    prompt = f"""
                    You are a Senior Growth Marketing Engineer.
                    REFERENCE DATA: {winners_context}
                    
                    TASK: Generate 10 Content Rows. 
                    Strategy: 7 Evolutionary (Structure-matched to Winners #1-5), 3 Revolutionary (New creative angles).
                    Context: Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                    
                    OUTPUT: Return a Markdown table with EXACTLY these columns: 
                    Usage Rank, New Content, Segmentation, Reference ID, Reference Content (Full), Hit Percentage, Reasoning.
                    """
                    
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering Complete")
                    
                    # Apply keyword highlighting to the final output
                    final_html = highlight_keywords(response.text, keywords_input)
                    st.markdown(final_html, unsafe_allow_html=True)
                    
                    st.download_button("📥 Download Analysis", response.text, file_name="engineered_content_audit.txt")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 To begin, upload your campaign CSV files to identify performance benchmarks.")        st.write("**Statistical Weighting:**")
        st.latex(r"Score = (CTR \times 0.7) + (Volume \times 0.3)")
        st.caption("""
        Prevents 'fluke' winners. We prioritize high-CTR creative only if it 
        has maintained performance over a significant audience scale.
        """)

    with st.expander("2. 7+3 Engineering Strategy", expanded=True):
        st.write("**Evolutionary (7 Rows):**")
        st.caption("Maps the 'Winning Skeleton' (emoji use, hook type, syntax) and swaps in new keywords.")
        st.write("**Revolutionary (3 Rows):**")
        st.caption("Pivots to new psychological angles (Social Proof, Scarcity, etc.) while keeping quality standards.")

    with st.expander("3. Validation & Mapping", expanded=True):
        st.write("**Hit % Estimation:**")
        st.caption("AI self-audit comparing structural alignment with the specific Reference ID.")
        st.write("**Semantic Highlighting:**")
        st.caption("Regex-powered visual confirmation of keyword integration.")

    st.divider()
    st.header("🎯 Campaign Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale")
    prod_description = st.text_area("Product Details", height=100)
    intention = st.text_input("Primary Goal", placeholder="e.g. Maximize CTR")

# --- 3. MAIN PAGE: DATA & OUTPUT ---
st.title("📊 Strategic Content Engineering")

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
            df['CTR_Clean'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Imp_Clean'] = pd.to_numeric(df[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            df['Power_Score'] = (df['CTR_Clean'] * 0.7) + ((df['Imp_Clean'] / df['Imp_Clean'].max()) * 0.3)
            winners = df.sort_values(by='Power_Score', ascending=False).head(10).copy()
        else:
            winners = df.head(10).copy()

        # Display Data
        t1, t2 = st.tabs(["📂 All Uploaded Data", "🏆 Performance Benchmarks (Winners)"])
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
                    TASK: Generate 10 Content Rows. 
                    Structure: 7 Evolutionary (Structure-matched), 3 Revolutionary (Pivot angles).
                    Context: Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                    
                    OUTPUT: Markdown table with columns: Usage Rank, New Content, Segmentation, Reference ID, Reference Content (Full), Hit Percentage, Reasoning.
                    """
                    
                    response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                    st.success("✅ Engineering Complete")
                    st.markdown(highlight_keywords(response.text, keywords_input), unsafe_allow_html=True)
                    st.download_button("📥 Download Export", response.text, file_name="engineered_content.txt")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload CSV files to begin the Performance-Weighted analysis.")
