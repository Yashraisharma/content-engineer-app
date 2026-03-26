import streamlit as st
import pandas as pd
from google import genai
import re
import io

# --- 1. CONFIG & SECURITY ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Strategic Edition", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    mark { border-radius: 4px; padding: 0 2px; }
    .status-box { padding: 20px; border-radius: 10px; background-color: #ffffff; border: 1px solid #e0e0e0; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 High-Fidelity Content Engineering")
st.write("Analyze performance benchmarks and engineer 10 optimized content variations.")

# --- 2. SIDEBAR: CONTEXT & GOALS ---
with st.sidebar:
    st.header("🎯 Strategy Parameters")
    keywords_input = st.text_input("Target Keywords", placeholder="e.g. BOGO, Sale, Limited")
    prod_description = st.text_area("Product/Offer Details", height=150)
    intention = st.text_input("Campaign Goal", placeholder="e.g. Higher CTR")
    
    st.divider()
    st.subheader("🛠️ Applied Logic")
    st.caption("• Weighted Ranking: 70% CTR / 30% Vol")
    st.caption("• Strategy: 7 Evolutionary + 3 Revolutionary")
    st.caption("• Model: Gemini 3 Flash (v2026)")

# --- 3. CORE UTILITIES ---
def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- 4. DATA ASSESSMENT ENGINE ---
uploaded_files = st.file_uploader("Upload Historical Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = [pd.read_csv(f) for f in uploaded_files]
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        
        # Column Identification
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)
        imp_col = col_map.get('impression', col_map.get('sent', col_map.get('delivered', None)))

        # Data Cleaning & Firm Logic Ranking
        if ctr_col and imp_col:
            df['CTR_Clean'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Imp_Clean'] = pd.to_numeric(df[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            
            # Ranking Logic: Efficiency (70%) + Reliability (30%)
            df['Power_Score'] = (df['CTR_Clean'] * 0.7) + ((df['Imp_Clean'] / df['Imp_Clean'].max()) * 0.3)
            winners = df.sort_values(by='Power_Score', ascending=False).head(10).copy()
        else:
            winners = df.head(10).copy()

        # UI: Show Processed Data
        tab1, tab2 = st.tabs(["📂 Full Uploaded Data", "🏆 Ranked Winners (References)"])
        
        with tab1:
            st.write(f"Showing all {len(df)} rows from uploaded files.")
            st.dataframe(df, use_container_width=True)
            
        with tab2:
            st.write("The following 10 rows were selected as the 'Mathematical Benchmarks' for engineering.")
            st.dataframe(winners[[content_col, ctr_col, imp_col, 'Power_Score']], use_container_width=True)

        # --- 5. AI ENGINEERING ---
        if st.button("🚀 Engineer 10 Variations & Rank Usage"):
            if not ACTIVE_KEY:
                st.error("Missing API Key! Please add 'GEMINI_API_KEY' to Streamlit Secrets.")
            else:
                with st.spinner("Analyzing structures and engineering 10 variations..."):
                    try:
                        client = genai.Client(api_key=ACTIVE_KEY)
                        winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                        
                        # Pack full, untruncated content for context
                        winners_context = ""
                        for _, row in winners.iterrows():
                            winners_context += f"--- {row['Ref_ID']} ---\nFULL CONTENT: {row[content_col]}\nCTR: {row.get(ctr_col, 'N/A')}\n\n"

                        prompt = f"""
                        REFERENCE DATA: {winners_context}
                        
                        TASK: Generate 10 Content Rows.
                        - 7 Evolutionary: Structurally identical to high-volume Winners.
                        - 3 Revolutionary: New creative angles using the same tone/quality.
                        
                        Parameters: Product: {prod_description} | Keywords: {keywords_input} | Goal: {intention}
                        
                        OUTPUT: A Markdown table with columns: 
                        1. **Usage Rank** (1-10, sorted by predicted performance)
                        2. **New Content**
                        3. **Segmentation**
                        4. **Reference ID**
                        5. **Reference Content (Full)**
                        6. **Hit Percentage**
                        7. **Reasoning**
                        """
                        
                        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                        
                        st.success("✅ Engineering Complete")
                        final_html = highlight_keywords(response.text, keywords_input)
                        st.markdown(final_html, unsafe_allow_html=True)
                        
                        # Download Options
                        st.download_button("📥 Download Analysis (.txt)", response.text, file_name="content_audit.txt")

                    except Exception as ai_err:
                        st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"Data Processing Error: {e}")
else:
    st.info("👋 To begin, upload your campaign CSV files in the box above.")
