import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def get_data():
        sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
        rows = []
        for s in sheets:
            df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all').reset_index(drop=True)
            for i in range(0, len(df), 2):
                r = df.iloc[i]
                if str(r.iloc[0]).lower() in ['city', 'category']: continue
                rows.append({'Name': str(r.iloc[0]).strip(), 'Total': int(r.iloc[1]), 'WA': int(r.iloc[7]), 'Push': int(r.iloc[3]), 'SMS': int(r.iloc[4]), 'Email': int(r.iloc[5])})
        return pd.DataFrame(rows)

    df = get_data()
    picks = st.session_state.selected_segments

    if not picks:
        st.warning("Select segments in Code X first!")
        picks = [st.selectbox("Or select one here manually:", df['Name'].unique())]

    # Combined Stats
    stats = df[df['Name'].isin(picks)].sum(numeric_only=True)

    st.subheader(f"🧬 Combined Stats: {', '.join(picks)}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Base", f"{int(stats['Total']):,}")
    c2.metric("WA (Karix)", f"{int(stats['WA']):,}")
    c3.metric("Push (Free)", f"{int(stats['Push']):,}")
    c4.metric("SMS (Vi)", f"{int(stats['SMS']):,}")

    # Financials
    st.divider()
    wa_rate = st.sidebar.number_input("WA Cost", value=0.78)
    aov = st.number_input("AOV (₹)", value=800)
    conv = st.slider("Conv %", 0.1, 5.0, 1.0)

    rev = (stats['WA'] * (conv/100)) * aov
    cost = (stats['WA'] - stats['Push']) * wa_rate
    st.success(f"**Combined Net Profit Lift:** ₹{int(rev - cost):,}")
