import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown("#### Logic Y: Volume-Based Planning with Vendor Costs")
    
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
                    'Free_Push_Reach': int(row_names.iloc[3]) if pd.notna(row_names.iloc[3]) else 0,
                    'Paid_WA_Reach': int(row_names.iloc[7]) if pd.notna(row_names.iloc[7]) else 0,
                    'Paid_SMS_Reach': int(row_names.iloc[4]) if pd.notna(row_names.iloc[4]) else 0
                })
            return pd.DataFrame(clean_data)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
            return None

    # 2. VENDOR PRICING INPUTS (Sidebar)
    st.sidebar.header("💳 Vendor Rates")
    wa_rate = st.sidebar.number_input("WhatsApp Rate (Karix)", value=0.78, step=0.01, help="Cost per marketing conversation")
    sms_rate = st.sidebar.number_input("SMS Rate (Vi)", value=0.13, step=0.01, help="Cost per DLT SMS unit")

    # 3. SELECTION UI
    view_type = st.radio("Targeting Dimension:", ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"], horizontal=True)
    df_active = load_and_stitch_volume(view_type)
    
    if df_active is not None and not df_active.empty:
        target_selection = st.selectbox(f"Select {view_type}", df_active['Name'].unique())
        data = df_active[df_active['Name'] == target_selection].iloc[0]
        
        st.divider()
        
        # 4. REACH METRICS
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Base", f"{data['Total_Audience']:,}")
        m2.metric("FREE Reach (Push)", f"{data['Free_Push_Reach']:,}")
        
        paid_lift = data['Paid_WA_Reach'] - data['Free_Push_Reach']
        m3.metric("Paid Lift (WA)", f"{paid_lift:,}", delta_color="normal")

        # 5. BUSINESS IMPACT PREDICTOR
        st.subheader("🔮 Campaign ROI Forecast")
        p1, p2 = st.columns(2)
        conv_rate = p1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0, 0.1)
        aov = p2.number_input("Average Order Value (₹)", value=800)

        # MATH LOGIC
        # Revenue
        total_rev = (data['Paid_WA_Reach'] * (conv_rate / 100)) * aov
        free_rev = (data['Free_Push_Reach'] * (conv_rate / 100)) * aov
        gross_incremental_rev = total_rev - free_rev
        
        # Costs
        wa_cost = paid_lift * wa_rate
        net_profit_lift = gross_incremental_rev - wa_cost
        roi = gross_incremental_rev / wa_cost if wa_cost > 0 else 0

        # DISPLAY RESULTS
        st.divider()
        res1, res2, res3 = st.columns(3)
        
        res1.write("**Financial Summary**")
        res1.info(f"Gross Incremental Revenue: **₹{int(gross_incremental_rev):,}**")
        
        res2.write("**Marketing Spend (Karix)**")
        res2.warning(f"Estimated WA Cost: **₹{int(wa_cost):,}**")
        
        res3.write("**Net Incremental Profit**")
        res3.success(f"**₹{int(net_profit_lift):,}**")

        st.write(f"### 📊 Strategy Verdict")
        if roi > 5:
            st.write(f"✅ **High ROI ({roi:.1f}x):** This campaign is highly profitable. Proceed with WhatsApp via Karix.")
        elif roi > 2:
            st.write(f"⚠️ **Moderate ROI ({roi:.1f}x):** Profitable, but consider A/B testing copy in Code X first.")
        else:
            st.write(f"❌ **Low ROI ({roi:.1f}x):** Cost is too high relative to gain. Stick to Free Push or SMS via Vi.")
