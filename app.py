import streamlit as st
import pandas as pd
from google import genai
import re

# --- SECURE CONFIG ---
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")

# Fix: Prevent Pandas from truncating long text in the reference strings
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 High-Fidelity Content Engineering")

with st.sidebar:
    st.header("📝 Campaign Context")
    keywords_input = st.text_input("Key Words", placeholder="e.g. Sale, New, Organic")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR")
    st.info("System will generate 7 similar & 3 brand new variations.")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = []
        for file in uploaded_files:
            file.seek(0)
            all_dfs.append(pd.read_csv(file))
        
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)

        st.subheader("📋 Source Data Preview")
        st.dataframe(df.head(5))

        if st.button("🚀 Run 10-Row Engineering"):
            if not ACTIVE_KEY:
                st.error("Missing API Key in Secrets!")
            else:
                with st.spinner("Analyzing all CSV data for 10 optimized variations..."):
                    try:
                        client = genai.Client(api_key=ACTIVE_KEY)
                        
                        # Identify Top Winners from across ALL uploaded files
                        if ctr_col:
                            df['CTR_Num'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce')
                            winners = df.sort_values(by='CTR_Num', ascending=False).head(10).copy()
                        else:
                            winners = df.head(10).copy()
                        
                        winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                        
                        # Build a massive, non-truncated context of ALL historical data
                        full_context = ""
                        for _, row in winners.iterrows():
                            full_context += f"--- {row['Ref_ID']} ---\nFULL CONTENT: {row[content_col]}\nORIGINAL CTR: {row.get(ctr_col, 'N/A')}\n\n"

                        response = client.models.generate_content(
                            model="gemini-3-flash-preview",
                            contents=f"""
                            You are a Senior Growth Marketing Strategist. 
                            
                            REFERENCE DATA (Top Performers):
                            {full_context}
                            
                            GOAL:
                            1. Keywords: {keywords_input}
                            2. Product: {prod_description}
                            3. Objective: {intention}
                            
                            REQUIRED OUTPUT (Exactly 10 Rows):
                            Rows 1-7: 'Evolutionary' - Closely follow the structure, tone, and length of the best performing References.
                            Rows 8-10: 'Revolutionary' - Entirely new creative angles that still respect the quality standards of the references.
                            
                            The 'Reference Content' column MUST contain the full, unedited text of the original row used for inspiration.
                            
                            FORMAT: Markdown table with 6 columns:
                            | New Content | Segmentation | Reference ID | Reference Content | Hit Percentage | Reasoning |
                            """
                        )
                        
                        st.success("✅ 10 Variations Engineered Successfully!")
                        highlighted_output = highlight_keywords(response.text, keywords_input)
                        st.markdown(highlighted_output, unsafe_allow_html=True)

                    except Exception as ai_err:
                        st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.info("Upload your campaign CSVs to begin the deep engineering process.")
