import streamlit as st
import pandas as pd
from google import genai
import re

# --- AUTO-CONFIG ---
API_KEY = "AIzaSyBPxucILyrIELye2IxFmLO_Ll1NRY2-LGM"

# 1. Page Configuration
st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")
st.info("Now featuring: Winner Reference Tracking & Keyword Highlighting")

# 2. Sidebar for Content Inputs
with st.sidebar:
    st.header("📝 Content Context")
    keywords_input = st.text_input("Key Words", placeholder="e.g. Sale, New, Organic")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR")

# Helper function to highlight keywords in the final text
def highlight_keywords(text, keywords_str):
    if not keywords_str:
        return text
    # Split keywords by comma or space and clean them
    kws = [k.strip() for k in re.split(r'[,\s]+', keywords_str) if k.strip()]
    for kw in kws:
        # Case-insensitive replacement with a yellow background and bold text
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(f'<mark style="background-color: #FFFF00; color: black; font-weight: bold;">{kw}</mark>', text)
    return text

# 3. File Processing
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        df = pd.concat([pd.read_csv(file) for file in uploaded_files], ignore_index=True)
        df.columns = df.columns.str.strip()
        
        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df.head(10))

        if st.button("🚀 Run Assessment & Engineering"):
            with st.spinner("Gemini 3 is mapping and highlighting..."):
                try:
                    client = genai.Client(api_key=API_KEY)
                    
                    # Selection Logic: Identify the Top 5 Winners
                    if 'CTR' in df.columns:
                        df['CTR_Num'] = pd.to_numeric(df['CTR'].astype(str).str.replace('%', ''), errors='coerce')
                        winners = df.sort_values(by='CTR_Num', ascending=False).head(5)
                        winners['Ref_ID'] = [f"Winner #{i+1}" for i in range(len(winners))]
                        winners_str = winners[['Ref_ID', 'Content', 'Segmentation', 'CTR']].to_string()
                    else:
                        winners = df.head(5)
                        winners['Ref_ID'] = [f"Row #{i+1}" for i in range(len(winners))]
                        winners_str = winners.to_string()

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"""
                        You are a Growth Marketing Specialist. 
                        
                        HISTORICAL WINNERS:
                        {winners_str}
                        
                        TASK:
                        Engineer 5 new content rows.
                        - Keywords: {keywords_input}
                        - Product: {prod_description}
                        - Goal: {intention}
                        
                        OUTPUT FORMAT:
                        Return a table with 4 columns: Content, Segmentation, Data-Driven Reasoning, Reference.
                        """
                    )
                    
                    st.success("Analysis Complete!")
                    
                    # Apply highlighting to the AI's response text
                    highlighted_output = highlight_keywords(response.text, keywords_input)
                    
                    # Display using unsafe_allow_html to render the <mark> tags
                    st.markdown(highlighted_output, unsafe_allow_html=True)

                except Exception as ai_err:
                    st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.warning("Please upload a CSV to begin.")
