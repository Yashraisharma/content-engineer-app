import streamlit as st
import pandas as pd
from google import genai
import re
import io

# --- 1. SECURE CONFIG ---
# Ensure "GEMINI_API_KEY" is added to your Streamlit Cloud Secrets
ACTIVE_KEY = st.secrets.get("GEMINI_API_KEY", "")
pd.set_option('display.max_colwidth', None)

st.set_page_config(page_title="Content Engineer Pro | Firm Logic", layout="wide")

# Custom CSS for better table visibility
st.markdown("""
    <style>
    .reportview-container .main .block-container{ max-width: 95%; }
    mark { border-radius: 4px; padding: 0 2px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Strategic Content Engineering & Analysis")
st.caption("2026 Edition: Powered by Gemini 3 Flash & Weighted Performance Logic")

# --- 2. SIDEBAR CONTEXT ---
with st.sidebar:
    st.header("🎯 Campaign Goals")
    keywords_input = st.text_input("Target Key Words", placeholder="e.g. BOGO, 50% OFF, Essentials")
    prod_description = st.text_area("Product/Offer Description", height=150)
    intention = st.text_input("Primary Intention", placeholder="e.g. Boost CTR / Drive App Installs")
    
    st.divider()
    st.subheader("⚙️ Engineering Strategy")
    st.write("✅ **7 Evolutionary:** Structure-matched to top winners.")
    st.write("🚀 **3 Revolutionary:** High-impact creative pivots.")

# --- 3. HELPER FUNCTIONS ---
def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# --- 4. DATA PROCESSING & FIRM LOGIC ---
uploaded_files = st.file_uploader("Upload Campaign CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = []
        for file in uploaded_files:
            file.seek(0)
            all_dfs.append(pd.read_csv(file))
        
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        
        # Column Mapping
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)
        imp_col = col_map.get('impression', col_map.get('sent', col_map.get('delivered', None)))

        if ctr_col and imp_col:
            # Clean and normalize data for Ranking
            df['CTR_Raw'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce') / 100
            df['Imp_Raw'] = pd.to_numeric(df[imp_col].astype(str).str.replace(',', ''), errors='coerce')
            
            # FIRM LOGIC FORMULA:
            # (CTR Weight: 0.7) + (Impression Volume Weight: 0.3)
            # We divide Impressions by the max in the set to normalize the scale.
            df['Power_Score'] = (df['CTR_Raw'] * 0.7) + ((df['Imp_Raw'] / df['Imp_Raw'].max()) * 0.3)
            winners = df.sort_values(by='Power_Score', ascending=False).head(10).copy()
        else:
            st.warning("⚠️ Could not find both CTR and Impression columns. Ranking will be based on file order.")
            winners = df.head(10).copy()

        st.subheader("📋 Top 10 Performance References (Ranked by Firm Logic)")
        st.dataframe(winners[[content_col, ctr_col, imp_col]].head(10), use_container_width=True)

        # --- 5. AI GENERATION ---
        if st.button("🚀 Engineer 10 High-Confidence Variations"):
            if not ACTIVE_KEY:
                st.error("API Key not found in Secrets. Please check Streamlit settings.")
            else:
                with st.spinner("Analyzing winning patterns and engineering content..."):
                    try:
                        client = genai.Client(api_key=ACTIVE_KEY)
                        winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                        
                        # Pack the full, untruncated content for the AI
                        winners_context = ""
                        for _, row in winners.iterrows():
                            winners_context += f"--- {row['Ref_ID']} ---\nFULL CONTENT: {row[content_col]}\nCTR: {row.get(ctr_col, 'N/A')}\n\n"

                        prompt = f"""
                        You are a Senior Growth Marketing Engineer. 
                        
                        HISTORICAL PERFORMANCE DATA:
                        {winners_context}
                        
                        NEW CAMPAIGN PARAMETERS:
                        - Product: {prod_description}
                        - Keywords: {keywords_input}
                        - Goal: {intention}
                        
                        INSTRUCTIONS:
                        1. Create EXACTLY 10 rows of new content.
                        2. Rows 1-7 (Evolutionary): Mimic the successful structure, emoji use, and 'hook' style of Winner #1 through #5.
                        3. Rows 8-10 (Revolutionary): Use the same high-quality standards but pivot to entirely new creative angles.
                        4. DO NOT truncate the 'Reference Content' column. Provide the FULL original text.
                        5. 'Hit Percentage' should reflect how well the new copy aligns with the successful 'Winning' pattern of its specific Reference ID.
                        
                        OUTPUT FORMAT: Return a Markdown table with these columns:
                        | New Content | Segmentation | Reference ID | Reference Content | Hit Percentage | Reasoning |
                        """
                        
                        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
                        
                        st.success("✅ Content Engineering Complete")
                        
                        # Render with Keyword Highlighting
                        final_html = highlight_keywords(response.text, keywords_input)
                        st.markdown(final_html, unsafe_allow_html=True)
                        
                        # --- 6. EXPORT ---
                        st.divider()
                        st.download_button(
                            label="📥 Download Results (.txt)",
                            data=response.text,
                            file_name=f"engineered_content_{intention.replace(' ', '_')}.txt",
                            mime="text/plain"
                        )

                    except Exception as ai_err:
                        st.error(f"AI Generation Error: {ai_err}")
    except Exception as e:
        st.error(f"Data Processing Error: {e}")
else:
    st.info("👋 Welcome! Upload your campaign CSVs to analyze your best performers and engineer 10 new high-CTR variations.")
