import streamlit as st
import pandas as pd
from google import genai  # The NEW 2026 library

# --- AUTO-CONFIG ---
API_KEY = "AIzaSyBPxucILyrIELye2IxFmLO_Ll1NRY2-LGM"

# 1. Page Configuration
st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")
st.info("Powered by Gemini 3 Flash")

# 2. Sidebar for Content Inputs
with st.sidebar:
    st.header("📝 Content Context")
    keywords = st.text_input("Key Words", placeholder="e.g. Sale, New, Organic")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR")

# 3. File Processing
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        # Load and clean data
        df = pd.concat([pd.read_csv(file) for file in uploaded_files], ignore_index=True)
        df.columns = df.columns.str.strip()
        
        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df.head(10))

        if st.button("🚀 Run Assessment & Engineering"):
            with st.spinner("Gemini 3 is analyzing patterns..."):
                try:
                    # Initialize the new 2026 Client
                    client = genai.Client(api_key=API_KEY)
                    
                    # Selection Logic: Find top performers
                    if 'CTR' in df.columns:
                        df['CTR_Num'] = pd.to_numeric(df['CTR'].astype(str).str.replace('%', ''), errors='coerce')
                        winners = df.sort_values(by='CTR_Num', ascending=False).head(5).to_string()
                    else:
                        winners = df.head(5).to_string()

                    # Call the Gemini 3 Flash Model
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=f"""
                        Assess these winning content rows:
                        {winners}
                        
                        Now, engineer 5 new content pieces based on those patterns.
                        - Keywords: {keywords}
                        - Product: {prod_description}
                        - Goal: {intention}
                        
                        Return a table with: Content, Segmentation, and Data-Driven Reasoning.
                        """
                    )
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.text)

                except Exception as ai_err:
                    st.error(f"AI Error: {ai_err}")
    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.warning("Please upload a CSV to begin.")
