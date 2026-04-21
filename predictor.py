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
                if str(r.iloc[0]).lower() in ['city', 'category', 'segment']: continue
                rows.append({
                    'Name': str(r.iloc[0]).strip(), 
                    'Total': int(r.iloc[1]), 
                    'WA': int(r.iloc[7]), 
                    'Push': int(r.iloc[3]), 
                    'SMS': int(r.iloc[4]), 
                    'Email': int(r.iloc[5])
                })
        return pd.DataFrame(rows)

    df_master = get_data()
    picks = st.session_state.get("selected_segments", [])

    if not picks:
        st.warning("⚠️ No segments selected in Code X. Select a segment below manually:")
        target_list = df_master['Name'].unique()
        picks = [st.selectbox("Select Segment", target_list)]

    # --- RESTORED COHORT DNA DASHBOARD ---
    combined_data = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader(f"🧬 Combined DNA: {', '.join(picks)}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Base", f"{int(combined_data['Total']):,}")
    c2.metric("WhatsApp", f"{int(combined_data['WA']):,}")
    c3.metric("Mobile Push", f"{int(combined_data['Push']):,}")
    c4.metric("SMS", f"{int(combined_data['SMS']):,}")
    c5.metric("Email", f"{int(combined_data['Email']):,}")

    # --- RESTORED ROI FORECAST PARAMETERS ---
    st.divider()
    st.subheader("🔮 Forecasting Parameters")
    f1, f2 = st.columns(2)
    conv = f1.slider("Expected Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    # --- RESTORED VENDOR PRICING ---
    st.sidebar.header("💳 Vendor Rates")
    wa_rate = st.sidebar.number_input("WhatsApp (Karix)", value=0.78)
    sms_rate = st.sidebar.number_input("SMS (Vi)", value=0.13)
    email_rate = st.sidebar.number_input("Email", value=0.03)

    # --- RESTORED FULL CHANNEL COMPARISON TABLE ---
    st.subheader("📊 Full Channel ROI Analysis")
    
    def calc_row(name, reach, unit_cost):
        revenue = (reach * (conv / 100)) * aov
        spend = reach * unit_cost
        profit = revenue - spend
        return {
            "Channel": name,
            "Reach": f"{int(reach):,}",
            "Spend": f"₹{int(spend):,}",
            "Revenue": f"₹{int(revenue):,}",
            "Net Profit": f"₹{int(profit):,}",
            "ROI": f"{(revenue/spend):.1f}x" if spend > 0 else "∞"
        }

    comparison_data = [
        calc_row("Mobile Push", combined_data['Push'], 0.0),
        calc_row("WhatsApp (Karix)", combined_data['WA'], wa_rate),
        calc_row("SMS (Vi)", combined_data['SMS'], sms_rate),
        calc_row("Email", combined_data['Email'], email_rate)
    ]
    st.table(pd.DataFrame(comparison_data))

    # --- RESTORED STRATEGY VERDICT ---
    st.divider()
    paid_lift_users = combined_data['WA'] - combined_data['Push']
    lift_revenue = (paid_lift_users * (conv/100)) * aov
    lift_cost = paid_lift_users * wa_rate
    
    st.write("### 💡 Final Growth Verdict")
    if lift_revenue > (lift_cost * 3):
        st.success(f"**PROCEED:** WhatsApp lift adds **₹{int(lift_revenue):,}** in revenue vs **₹{int(lift_cost):,}** spend.")
    else:
        st.warning("**CAUTION:** Marginal ROI. Consider using **SMS (Vi)** or strictly free **Mobile Push**.")
