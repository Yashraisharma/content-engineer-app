import streamlit as st
import pandas as pd
import google.generativeai as genai

st.set_page_config(page_title="Content Engineer", layout="wide")
st.title("📊 Content Assessment & Engineering")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter Google API Key", type="password")
    keywords = st.text_input("Keywords")
    prod_description = st.text_area("Product Description")
    intention = st.text_input("Intention (e.g. Sales)")
    if api_key:
        genai.configure(api_key=api_key)

uploaded_files = st.file_uploader("Upload CSVs", type="csv", accept_multiple_files=True)

if uploaded_files:
    df = pd.concat([pd.read_csv(f) for f in uploaded_files], ignore_index=True)
    cols = ['Content', 'Segmentation', 'Sent', 'Viewed/IMP%', 'Impression', 'Click', 'CTR', 'Converted']
    df_filtered = df[[c for c in cols if c in df.columns]]
    st.dataframe(df_filtered.head())

    if st.button("Generate Content"):
        if not api_key: st.error("Add API Key")
        else:
            winners = df_filtered.sort_values(by='CTR', ascending=False).head(5).to_string()
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Data: {winners}\nKeywords: {keywords}\nProduct: {prod_description}\nGoal: {intention}\nGenerate 5 new rows of content in a table."
            response = model.generate_content(prompt)
            st.write(response.text)
