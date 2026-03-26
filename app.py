import streamlit as st
import pandas as pd
from google import genai
import re

# --- AUTO-CONFIG ---
API_KEY = "AIzaSyBPxucILyrIELye2IxFmLO_Ll1NRY2-LGM"

st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")

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

uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        all_dfs = []
        for file in uploaded_files:
            file.seek(0)
            all_dfs.append(pd.read_csv(file))
        
        df = pd.concat(all_dfs, ignore_index=True)
        df.columns = df.columns.str.strip()
        
        # Mapping columns for stability
        col_map = {c.lower(): c for c in df.columns}
        content_col = col_map.get('content', df.columns[0])
        ctr_col = col_map.get('ctr', None)

        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df.head(10))

        if st.button("🚀 Run Deep Assessment"):
            with st.spinner("Gemini 3 is calculating Hit Percentages and Mapping Content..."):
                try:
                    client = genai.Client(api_key=API_KEY)
                    
                    # Identify Top 5 Winners
                    if ctr_col:
                        df['CTR_Num'] = pd.to_numeric(df[ctr_col].astype(str).str.replace('%', ''), errors='coerce')
                        winners = df.sort_values(by='CTR_Num', ascending=False).head(5).copy()
                    else:
                        winners = df.head(5).copy()
                    
                    winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                    
                    # Create a detailed context string including the actual content of winners
                    winners_context = ""
                    for _, row in winners.iterrows():
                        winners_context += f"[{row['Ref_ID']}] Content: {row[content_col]} | CTR: {row.get(ctr_col, 'N/A')}\n"

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"""
                        You are a Growth Marketing Analyst. 
                        
                        HISTORICAL DATA (REFERENCES):
                        {winners_context}
                        
                        TASK:
                        Engineer 5 new content rows.
                        - Keywords: {keywords_input}
                        - Product: {prod_description}
                        - Goal: {intention}
                        
                        OUTPUT FORMAT:
                        Return a table with these EXACT 6 columns:
                        1. **New Content**: The engineered text.
                        2. **Segmentation**: Target audience.
                        3. **Reference ID**: The Ref_ID used (e.g. Winner #1).
                        4. **Reference Content**: The original content from that Ref_ID.
                        5. **Hit Percentage**: An estimated percentage (e.g. 85%) of how effectively the new content retains the 'winning' structure of the reference while hitting the new keywords.
                        6. **Reasoning**: Why this specific change was made.
                        """
                    )
                    
                    st.success("Analysis Complete!")
                    highlighted_output = highlight_keywords(response.text, keywords_input)
                    st.markdown(highlighted_output, unsafe_allow_html=True)

                except Exception as ai_err:
                    st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.warning("Please upload a CSV to begin.")
