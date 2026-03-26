import streamlit as st
import pandas as pd
from google import genai
import re

# --- AUTO-CONFIG ---
API_KEY = "AIzaSyBPxucILyrIELye2IxFmLO_Ll1NRY2-LGM"

st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")

# 1. Sidebar Setup
with st.sidebar:
    st.header("📝 Content Context")
    keywords_input = st.text_input("Key Words", placeholder="e.g. Sale, New, Organic")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR")

def highlight_keywords(text, keywords_str):
    if not keywords_str: return text
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# 2. File Upload & Normalization
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = []
        for file in uploaded_files:
            file.seek(0) # Reset file pointer to avoid "EmptyDataError"
            temp_df = pd.read_csv(file)
            # CLEAN HEADERS: Remove spaces and make lowercase for internal checking
            temp_df.columns = temp_df.columns.str.strip()
            all_dfs.append(temp_df)
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # SMART COLUMN MAPPING
        # This finds 'Content' even if it's 'content' or ' CONTENT'
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0]) # Default to 1st col if not found
        ctr_col = col_map.get('ctr', None)

        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df.head(10))

        if st.button("🚀 Run Assessment & Engineering"):
            with st.spinner("Gemini 3 is analyzing..."):
                try:
                    client = genai.Client(api_key=API_KEY)
                    
                    # Selection Logic using the mapped columns
                    if ctr_col:
                        df['CTR_Num'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce')
                        winners = df.sort_values(by='CTR_Num', ascending=False).head(5)
                    else:
                        winners = df.head(5)
                    
                    winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                    # Only send relevant columns to the AI to save tokens
                    winners_context = winners[['Ref_ID', content_col]].to_string()

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"""
                        HISTORICAL WINNERS:
                        {winners_context}
                        
                        TASK:
                        Engineer 5 new content rows.
                        - Keywords: {keywords_input}
                        - Product: {prod_description}
                        - Goal: {intention}
                        
                        OUTPUT FORMAT:
                        Return a table with: Content, Segmentation, Data-Driven Reasoning, Reference.
                        """
                    )
                    
                    st.success("Analysis Complete!")
                    highlighted_output = highlight_keywords(response.text, keywords_input)
                    st.markdown(highlighted_output, unsafe_allow_html=True)

                except Exception as ai_err:
                    st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"Data Processing Error: {e}")
else:
    st.warning("Please upload a CSV to begin.")
