import streamlit as st
import pandas as pd
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")

# 2. Sidebar Setup
with st.sidebar:
    st.header("🔑 API Setup")
    # Using a text input for the key
    user_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    st.header("📝 Content Context")
    keywords = st.text_input("Key Words", placeholder="e.g. Sale, New, Organic")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR")

# 3. File Processing
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        df = pd.concat([pd.read_csv(file) for file in uploaded_files], ignore_index=True)
        
        # Standardize column names (removing spaces/caps issues)
        df.columns = df.columns.str.strip()
        
        required_cols = ['Content', 'Segmentation', 'Sent', 'Viewed/IMP%', 'Impression', 'Click', 'CTR', 'Converted']
        existing_cols = [c for c in required_cols if c in df.columns]
        df_filtered = df[existing_cols]

        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df_filtered.head(10))

        if st.button("🚀 Run Assessment & Engineering"):
            if not user_key:
                st.error("Please enter your API Key in the sidebar!")
            else:
                with st.spinner("AI is analyzing..."):
                    try:
                        genai.configure(api_key=user_key)
                        
                        # Fix CTR for sorting (handles '5%' or '0.05')
                        if 'CTR' in df_filtered.columns:
                            temp_df = df_filtered.copy()
                            temp_df['CTR_Numeric'] = pd.to_numeric(temp_df['CTR'].astype(str).str.replace('%', ''), errors='coerce')
                            winners = temp_df.sort_values(by='CTR_Numeric', ascending=False).head(5).drop(columns=['CTR_Numeric']).to_string()
                        else:
                            winners = df_filtered.head(5).to_string()

                        # FIX: Using the full model path to avoid 404
                        model = genai.GenerativeModel('models/gemini-1.5-flash')
                        
                        prompt = f"""
                        Historical Data (Winners):
                        {winners}
                        
                        New Task:
                        Keywords: {keywords}
                        Product: {prod_description}
                        Goal: {intention}
                        
                        Generate 5 new content rows. Provide a table with: Content, Segmentation, and Reasoning.
                        """
                        
                        response = model.generate_content(prompt)
                        st.success("Success!")
                        st.markdown(response.text)

                    except Exception as ai_err:
                        st.error(f"AI Connection Error: {ai_err}")
    except Exception as e:
        st.error(f"File Error: {e}")
else:
    st.warning("Upload a CSV to start.")
