import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    # --- 1. CLOCK & CALENDAR ---
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    current_hour = now.hour
    
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown(f"#### Logic Y | {current_date}")
    
    # 2. DATA SOURCE
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def get_data():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            rows = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all').reset_index(drop=True)
                for i in range(0, len(df), 2):
                    r = df.iloc[i]
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment', 'status']: continue
                    rows.append({
                        'Name': str(r.iloc[0]).strip(), 
                        'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 
                        'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0, 
                        'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0
                    })
            return pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Excel Load Error: {e}")
            return pd.DataFrame()

    df_master = get_data()
    # Pull selections from Code X memory
    picks = st.session_state.get("selected_segments", [])

    if not picks:
        st.warning("⚠️ No segments selected in Code X. Select a segment below for manual analysis:")
        target_list = df_master['Name'].unique() if not df_master.empty else ["Hyderabad"]
        picks = [st.selectbox("Select Segment", target_list)]

    # --- 3. LIVE CONTEXTUAL INTELLIGENCE BOX ---
    st.divider()
    primary_city = picks[0] if picks else "Hyderabad"
    
    # Dynamic logic for News/Weather (Simulated for April 21, 2026)
    weather_status = "🌡️ 41°C - Severe Heatwave Alert"
    news_ticker = "📢 Apollo 247 App crosses new milestone in chronic care registrations."
    
    # Demographic mapping based on your segments
    demo_map = {
        "Hyderabad": {"old": "12%", "moms": "4.5%", "tech": "89%", "type": "Tier 1 / IT Hub"},
        "Delhi": {"old": "15%", "moms": "3.2%", "tech": "92%", "type": "Tier 1 / Metro"},
        "Cardio_Diab": {"old": "68%", "moms": "1.2%", "tech": "42%", "type": "Chronic Patient Segment"},
        "Circle_Active": {"old": "22%", "moms": "8.5%", "tech": "98%", "type": "Loyalty Program"}
    }
    intel = demo_map.get(primary_city, {"old": "15%", "moms": "5%", "tech": "75%", "type": "General Base"})

    st.markdown(f"""
        <div style="background-color: #f0f9ff; border-left: 5px solid #0ea5e9; padding: 15px; border-radius: 5px;">
            <p style="margin: 0; color: #0c4a6e;"><b>📍 Segment Intel: {primary_city}</b></p>
            <p style="margin: 5px 0; font-size: 0.9em;"><b>Weather:</b> {weather_status} | <b>News:</b> {news_ticker}</p>
            <hr style="margin: 10px 0; border: 0; border-top: 1px solid #bae6fd;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85em;">
                <span>👴 Seniors: <b>{intel['old']}</b></span>
                <span>🍼 New Moms: <b>{intel['moms']}</b></span>
                <span>📱 Tech Capable: <b>{intel['tech']}</b></span>
                <span>🏷️ Type: <b>{intel['type']}</b></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 4. CROSS-SELL RECOMMENDATIONS (Apollo Pharmacy) ---
    st.markdown("### 🛒 Smart Cross-Sell Recommendations")
    
    # Logic: If it's hot and it's morning, sell hydration/sunscreen.
    # If it's a chronic segment, sell specialized care.
    if "Cardio" in primary_city or "Diab" in primary_city:
        p1_name, p1_link = "Apollo Pharmacy Digital Blood Pressure Monitor", "https://www.apollopharmacy.in/shop-by-category/health-devices"
        p2_name, p2_link = "Apollo Pharmacy Sugar-Free Protein Powder", "https://www.apollopharmacy.in/shop-by-category/diabetes-care"
    elif current_hour < 12: # Morning logic
        p1_name, p1_link = "ORSL Rehydrate Apple Drink (Heatwave Shield)", "https://www.apollopharmacy.in/shop-by-category/otc"
        p2_name, p2_link = "Apollo Pharmacy SPF 50 Sunscreen Lotion", "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care"
    else:
        p1_name, p1_link = "Apollo Life Tulsi & Ginger Herbal Tea", "https://www.apollopharmacy.in/shop-by-category/health-drinks"
        p2_name, p2_link = "Sensodyne Whitening Toothpaste (Bundle)", "https://www.apollopharmacy.in/shop-by-category/personal-care"

    cs1, cs2 = st.columns(2)
    cs1.info(f"**Primary Push Suggestion:**\n{p1_name}")
    cs1.markdown(f"[Buy on Apollo Pharmacy]({p1_link})")
    cs2.success(f"**Cross-Sell Upsell:**\n{p2_name}")
    cs2.markdown(f"[Buy on Apollo Pharmacy]({p2_link})")

    # --- 5. COHORT DNA SUMMARY ---
    st.divider()
    combined_data = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Segment Reach DNA")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Base", f"{int(combined_data['Total']):,}")
    s2.metric("WhatsApp", f"{int(combined_data['WA']):,}")
    s3.metric("Mobile Push", f"{int(combined_data['Push']):,}")
    s4.metric("SMS", f"{int(combined_data['SMS']):,}")
    s5.metric("Email", f"{int(combined_data['Email']):,}")

    # --- 6. ROI ANALYSIS TABLE ---
    st.divider()
    st.sidebar.header("💳 Vendor Rates")
    wa_rate = st.sidebar.number_input("WA (Karix)", value=0.78)
    sms_rate = st.sidebar.number_input("SMS (Vi)", value=0.13)
    
    st.subheader("🔮 Forecasting & ROI")
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("AOV (₹)", value=800)

    def calc_row(name, reach, unit_cost):
        rev = (reach * (conv / 100)) * aov
        spend = reach * unit_cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "Profit": f"₹{int(rev-spend):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    table_data = [
        calc_row("Mobile Push", combined_data['Push'], 0.0),
        calc_row("WhatsApp (Karix)", combined_data['WA'], wa_rate),
        calc_row("SMS (Vi)", combined_data['SMS'], sms_rate)
    ]
    st.table(pd.DataFrame(table_data))

    # --- 7. STRATEGIC VERDICT ---
    lift_users = combined_data['WA'] - combined_data['Push']
    lift_rev = (lift_users * (conv/100)) * aov
    lift_cost = lift_users * wa_rate
    
    st.write("### 💡 Strategy Verdict")
    if lift_rev > (lift_cost * 3):
        st.success(f"**PROCEED:** WA adds **₹{int(lift_rev):,}** revenue vs **₹{int(lift_cost):,}** cost.")
    else:
        st.warning("**MARGINAL:** Consider using SMS or free Push to protect margins.")
