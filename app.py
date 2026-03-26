import streamlit as st
import pandas as pd
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="Content Engineer Pro", layout="wide")
st.title("📊 Content Assessment & Engineering")
st.info("Upload your CSVs, and let AI assess your best performers to create new content.")

# 2. Sidebar Setup
with st.sidebar:
    st.header("🔑 API Setup")
    api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get it free at aistudio.google.com")
    
    st.header("📝 Content Context")
    keywords = st.text_input("Key Words", placeholder="e.g. Discount, New, Organic")
    prod_description = st.text_area("Product Description", placeholder="What are we talking about?")
    intention = st.text_input("Intention", placeholder="e.g. Higher CTR, Sales, or Awareness")

# 3. File Processing
uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    try:
        # Merge all CSVs
        df_list = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(df_list, ignore_index=True)
        
        # Define required columns based on your request
        required_cols = ['Content', 'Segmentation', 'Sent', 'Viewed/IMP%', 'Impression', 'Click', 'CTR', 'Converted']
        
        # Only keep columns that exist in the uploaded files
        existing_cols = [c for c in required_cols if c in df.columns]
        df_filtered = df[existing_cols]

        st.subheader("📋 Data Assessment Preview")
        st.dataframe(df_filtered.head(10))

        # 4. The Generation Trigger
        if st.button("🚀 Run Assessment & Engineering"):
            if not api_key:
                st.error("Please enter your API Key in the sidebar!")
            else:
                with st.spinner("AI is analyzing your data..."):
                    try:
                        # Configure AI
                        genai.configure(api_key=api_key)
                        
                        # Selection Logic: Find top performers by CTR if available
                        if 'CTR' in df_filtered.columns:
                            # Clean CTR column (remove % if present and convert to float)
                            df_filtered['CTR_Score'] = df_filtered['CTR'].astype(str).str.replace('%', '').astype(float)
                            winners = df_filtered.sort_values(by='CTR_Score', ascending=False).head(5).drop(columns=['CTR_Score']).to_string()
                        else:
                            winners = df_filtered.head(5).to_string()

                        # Universal stable model name
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        prompt = f"""
                        You are a Growth Marketing Specialist. 
                        
                        STEP 1: ASSESS this historical data of top-performing content:
                        {winners}
                        
                        STEP 2: ENGINEER 5 new content rows based on the patterns found above.
                        - Use Keywords: {keywords}
                        - Product: {prod_description}
                        - Goal: {intention}
                        
                        FORMAT: Provide a table with columns: Content, Segmentation, and a brief 'Reasoning' for why this will work based on the data.
                        """
                        
                        response = model.generate_content(prompt)
                        
                        st.success("Analysis Complete!")
                        st.markdown("### 🎯 New Engineered Content")
                        st.write(response.text)

                    except Exception as ai_err:
                        st.error(f"AI Error: {ai_err}. Check if your API key is active or if the model name is correct.")
    
    except Exception as e:
        st.error(f"File Error: {e}. Ensure your CSVs are formatted correctly with headers.")

else:
    st.warning("Please upload at least one CSV file to begin.")
