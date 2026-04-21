import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown("#### Logic Y: Multi-Channel Attribution & Circle Intelligence")
    
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
                if str(row_names.iloc[0]).lower() in ['city', 'category', 'segment', 'status', 'sku_id']: 
                    continue
                
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
            st.error(f"⚠️ Data Error on sheet '{sheet_name}': {e}")
            return None

    # 2. VENDOR PRICING (Sidebar)
    st.sidebar.header("💳 Channel Unit Costs")
    wa_cost = st.sidebar.number_input("WhatsApp (Karix)", value=0.78, step=0.01)
    sms_cost = st.sidebar.number_input("SMS (Vi)", value=0.13, step=0.01)
    email_cost = st.sidebar.number_input("Email", value=0.03, step=0.01)
    push_cost = 0.00

    # 3. CORE SELECTION
    view_type = st.radio("Targeting Dimension:", ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment", "Circle_Activate_status"], horizontal=True)
    df_active = load_and_stitch_volume(view_type)
    
    if df_active is not None and not df_active.empty:
        target = st.selectbox(f"Select {view_type}", df_active['Name'].unique())
        data = df_active[df_active['Name'] == target].iloc[0]
        
        st.divider()

        # --- NEW SECTION: FULL COHORT STATS ---
        st.subheader(f"🧬 Cohort DNA: {target}")
        
        # Displaying ALL Raw Stats at once
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Base", f"{data['Total_Audience']:,}")
        s2.metric("WhatsApp", f"{data['WA_Raw']:,}")
        s3.metric("Mobile Push", f"{data['Mobile_Push_Raw']:,}")
        s4.metric("SMS", f"{data['SMS_Raw']:,}")
        s5.metric("Email", f"{data['Email_Raw']:,}")

        # 4. CONVERSION PARAMETERS
        st.divider()
        st.subheader("🔮 Forecasting Parameters")
        p1, p2 = st.columns(2)
        conv_rate = p1.slider("Expected Conversion Rate (%)", 0.1, 5.0, 1.0, 0.1)
        aov = p2.number_input("Average Order Value (₹)", value=800)

        # 5. FULL CHANNEL COMPARISON TABLE
        st.subheader("📊 Channel ROI Analysis")
        
        def calc_metrics(name, reach, unit_cost):
            revenue = (reach * (conv_rate / 100)) * aov
            spend = reach * unit_cost
            net_profit = revenue - spend
            roi = revenue / spend if spend > 0 else 0
            return {
                "Channel": name,
                "Reach (Headcount)": f"{int(reach):,}",
                "Est. Spend": f"₹{int(spend):,}",
                "Proj. Revenue": f"₹{int(revenue):,}",
                "Net Profit": f"₹{int(net_profit):,}",
                "ROI": f"{roi:.1f}x" if spend > 0 else "∞ (Free)"
            }

        comparison = [
            calc_metrics("Mobile Push", data['Mobile_Push_Raw'], push_cost),
            calc_metrics("WhatsApp (Karix)", data['WA_Raw'], wa_cost),
            calc_metrics("SMS (Vi)", data['SMS_Raw'], sms_cost),
            calc_metrics("Email", data['Email_Raw'], email_cost)
        ]
        st.table(pd.DataFrame(comparison))

        # 6. CIRCLE PENETRATION
        if view_type != "Circle_Activate_status":
            st.divider()
            st.subheader("⭕ Global Circle Context")
            df_c = load_and_stitch_volume("Circle_Activate_status")
            if df_c is not None:
                c_data = df_c.iloc[0] 
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Circle Members", f"{c_data['Total_Audience']:,}")
                c2.metric("Circle Push Reach", f"{c_data['Mobile_Push_Raw']:,}")
                c3.metric("Circle WA Reach", f"{c_data['WA_Raw']:,}")

        # 7. STRATEGIC VERDICT
        st.divider()
        paid_lift_users = data['WA_Raw'] - data['Mobile_Push_Raw']
        incremental_rev = (paid_lift_users * (conv_rate/100)) * aov
        wa_spend = paid_lift_users * wa_cost
        
        st.write("### 💡 Strategy Verdict")
        if incremental_rev > (wa_spend * 3):
            st.success(f"**PROCEED:** WhatsApp lift adds **₹{int(incremental_rev):,}** in revenue vs **₹{int(wa_spend):,}** spend.")
        else:
            st.warning("**CAUTION:** Marginal ROI. Stick to free **Mobile Push** or **SMS (Vi)** to protect margins.")
