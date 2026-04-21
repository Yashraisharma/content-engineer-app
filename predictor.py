import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown("#### Logic Y: Multi-Channel Volume & Cost Attribution")
    
    # 1. THE DATA SOURCE
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def load_and_stitch_volume(sheet_name):
        try:
            df = pd.read_excel(EXCEL_URL, sheet_name=sheet_name, engine='openpyxl').dropna(how='all').reset_index(drop=True)
            clean_data = []
            for i in range(0, len(df), 2):
                if i+1 >= len(df): break
                row_names = df.iloc[i]
                if str(row_names.iloc[0]).lower() in ['city', 'category', 'segment', 'status']: continue
                
                clean_data.append({
                    'Name': str(row_names.iloc[0]).strip(),
                    'Total_Audience': int(row_names.iloc[1]) if pd.notna(row_names.iloc[1]) else 0,
                    'Web_Push_Raw': int(row_names.iloc[2]) if pd.notna(row_names.iloc[2]) else 0,
                    'Mobile_Push_Raw': int(row_names.iloc[3]) if pd.notna(row_names.iloc[3]) else 0,
                    'SMS_Raw': int(row_names.iloc[4]) if pd.notna(row_names.iloc[4]) else 0,
                    'Email_Raw': int(row_names.iloc[5]) if pd.notna(row_names.iloc[5]) else 0,
                    'WA_Raw': int(row_names.iloc[7]) if pd.notna(row_names.iloc[7]) else 0
                })
            return pd.DataFrame(clean_data)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            return None

    # 2. VENDOR PRICING INPUTS (Sidebar)
    st.sidebar.header("💳 Channel Unit Costs")
    wa_cost_unit = st.sidebar.number_input("WhatsApp (Karix)", value=0.78, step=0.01)
    sms_cost_unit = st.sidebar.number_input("SMS (Vi)", value=0.13, step=0.01)
    email_cost_unit = st.sidebar.number_input("Email", value=0.03, step=0.01)
    push_cost_unit = st.sidebar.number_input("Mobile Push", value=0.00, disabled=True)

    # 3. SELECTION UI
    view_type = st.radio("Targeting Dimension:", ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"], horizontal=True)
    df_active = load_and_stitch_volume(view_type)
    
    if df_active is not None and not df_active.empty:
        target_selection = st.selectbox(f"Select {view_type}", df_active['Name'].unique())
        data = df_active[df_active['Name'] == target_selection].iloc[0]
        
        st.divider()

        # 4. CONVERSION INPUTS
        st.subheader("🔮 Forecasting Parameters")
        p1, p2 = st.columns(2)
        conv_rate = p1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0, 0.1)
        aov = p2.number_input("Average Order Value (₹)", value=800)

        # 5. FULL CHANNEL COMPARISON TABLE
        st.subheader("📊 Full Base Channel Analysis")
        
        def calc_row(name, reach, unit_cost):
            revenue = (reach * (conv_rate / 100)) * aov
            total_cost = reach * unit_cost
            net_profit = revenue - total_cost
            roi = revenue / total_cost if total_cost > 0 else float('inf')
            return {
                "Channel": name,
                "Reach (Headcount)": f"{int(reach):,}",
                "Est. Spend": f"₹{int(total_cost):,}",
                "Proj. Revenue": f"₹{int(revenue):,}",
                "Net Profit": f"₹{int(net_profit):,}",
                "ROI": f"{roi:.1f}x" if roi != float('inf') else "∞ (Free)"
            }

        comparison_data = [
            calc_row("Mobile Push", data['Mobile_Push_Raw'], push_cost_unit),
            calc_row("WhatsApp (Karix)", data['WA_Raw'], wa_cost_unit),
            calc_row("SMS (Vi)", data['SMS_Raw'], sms_cost_unit),
            calc_row("Email", data['Email_Raw'], email_cost_unit)
        ]
        
        st.table(pd.DataFrame(comparison_data))

        # 6. STRATEGIC SUMMARY
        st.divider()
        st.subheader("💡 Growth Strategy Verdict")
        
        total_max_reach = data['WA_Raw'] # Usually WA is the highest reach channel
        max_rev = (total_max_reach * (conv_rate/100)) * aov
        
        col1, col2 = st.columns(2)
        col1.info(f"**Total Addressable Base:** {data['Total_Audience']:,} users")
        col2.success(f"**Max Revenue Opportunity:** ₹{int(max_rev):,}")
        
        st.write(f"To reach the **full base** of {data['Total_Audience']:,} users, a multi-channel mix is required. Using **WhatsApp** provides the highest incremental lift over free channels, contributing to the projected profit shown above.")
