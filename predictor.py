import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    
    # 1. THE DATA ENGINE
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
            st.error(f"Error loading Excel: {e}")
            return pd.DataFrame()

    df_master = get_data()

    # --- 2. GLITCH FIX: SYNC SELECTION TO MEMORY ---
    def sync_segments():
        st.session_state.selected_segments = st.session_state.segment_widget_key

    # Initialize session state if empty
    if "selected_segments" not in st.session_state:
        st.session_state.selected_segments = []

    st.sidebar.markdown("### 🎯 Segment Controls")
    if st.sidebar.button("🗑️ Reset All Filters"):
        st.session_state.selected_segments = []
        st.rerun()

    # The Multiselect using a STABLE KEY and CALLBACK
    picks = st.multiselect(
        "🔍 Search & Analyze Segments:", 
        options=df_master['Name'].unique().tolist(),
        default=st.session_state.selected_segments,
        key="segment_widget_key",
        on_change=sync_segments
    )

    if not picks:
        st.info("👋 Select a segment from the search bar above to begin.")
        return

    # --- 3. DYNAMIC INTEL ENGINE (PRIORITY: MOM > CHRONIC > CITY) ---
    primary = picks[0].lower()
    
    if any(x in primary for x in ["mom", "baby", "infant", "pediatric", "mother"]):
        intel = {"old": "2%", "moms": "96%", "tech": "95%", "type": "Motherhood", "color": "#fdf2f8", "border": "#ec4899"}
        vibe = "🍼 New Parent / Baby Care"
        p1 = {"name": "Pampers All-Round Protection Diapers", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/diapers"}
        p2 = {"name": "Himalaya Baby Wipes (Pack of 3)", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/baby-wipes"}
    elif any(x in primary for x in ["cardio", "diab", "pharma", "chronic", "sugar", "bp"]):
        intel = {"old": "75%", "moms": "1%", "tech": "42%", "type": "Chronic Patient", "color": "#f0fdf4", "border": "#22c55e"}
        vibe = "💊 Chronic/Elderly Care"
        p1 = {"name": "Apollo Pharmacy Digital BP Monitor", "link": "https://www.apollopharmacy.in/shop-by-category/health-devices/bp-monitors"}
        p2 = {"name": "OneTouch Select Plus Glucometer Strips", "link": "https://www.apollopharmacy.in/shop-by-category/diabetes-care/test-strips"}
    else:
        intel = {"old": "18%", "moms": "6%", "tech": "91%", "type": "Urban Consumer", "color": "#eff6ff", "border": "#3b82f6"}
        vibe = "🏢 Urban / City Focused"
        p1 = {"name": "ORSL Rehydrate Apple (Heatwave Essential)", "link": "https://www.apollopharmacy.in/shop-by-category/otc"}
        p2 = {"name": "Apollo Pharmacy SPF 50 Sunscreen", "link": "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care"}

    # --- 4. HIGH-VISIBILITY INTELLIGENCE CARD ---
    st.markdown(f"""
        <div style="background-color: {intel['color']}; border: 2px solid {intel['border']}; padding: 20px; border-radius: 10px; color: #1e293b; margin-bottom: 25px;">
            <h3 style="margin-top: 0; color: #0f172a;">🕵️ Live Context: {picks[0]}</h3>
            <p style="font-weight: bold; font-size: 1.2em; color: {intel['border']}; margin-bottom: 15px;">{vibe} | {current_date}</p>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
                    <span style="font-size: 1.5em;">👵</span><br><b>Senior Citizens</b><br><span style="font-size: 1.2em;">{intel['old']}</span>
                </div>
                <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
                    <span style="font-size: 1.5em;">🍼</span><br><b>New Moms</b><br><span style="font-size: 1.2em;">{intel['moms']}</span>
                </div>
                <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
                    <span style="font-size: 1.5em;">📱</span><br><b>Smartphone Savvy</b><br><span style="font-size: 1.2em;">{intel['tech']}</span>
                </div>
                <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; text-align: center;">
                    <span style="font-size: 1.5em;">🏷️</span><br><b>Market Type</b><br><span style="font-size: 1.0em;">{intel['type']}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 5. SMART CROSS-SELL ---
    st.markdown("### 🛒 Targeted Cross-Sell Opportunities")
    xs1, xs2 = st.columns(2)
    xs1.info(f"**Primary Campaign Focus:**\n{p1['name']}")
    xs1.markdown(f"[Buy on Apollo Pharmacy]({p1['link']})")
    xs2.success(f"**Suggested Cross-Sell:**\n{p2['name']}")
    xs2.markdown(f"[Buy on Apollo Pharmacy]({p2['link']})")

    # --- 6. REACH DNA & ROI MATH ---
    st.divider()
    combined_data = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Reach DNA (Aggregated)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(combined_data['Total']):,}")
    m2.metric("WhatsApp", f"{int(combined_data['WA']):,}")
    m3.metric("Mobile Push", f"{int(combined_data['Push']):,}")
    m4.metric("SMS", f"{int(combined_data['SMS']):,}")
    m5.metric("Email", f"{int(combined_data['Email']):,}")

    st.divider()
    st.subheader("🔮 Campaign ROI Forecast")
    wa_rate = st.sidebar.number_input("WA Cost (Karix)", value=0.78)
    sms_rate = st.sidebar.number_input("SMS Cost (Vi)", value=0.13)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc_channel(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Rev": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    table = [
        calc_channel("Mobile Push", combined_data['Push'], 0.0),
        calc_channel("WhatsApp", combined_data['WA'], wa_rate),
        calc_channel("SMS", combined_data['SMS'], sms_rate)
    ]
    st.table(pd.DataFrame(table))

    # --- 7. STRATEGIC VERDICT ---
    lift_users = combined_data['WA'] - combined_data['Push']
    lift_rev = (lift_users * (conv/100)) * aov
    lift_cost = lift_users * wa_rate
    st.markdown(f"### 💡 Growth Verdict")
    if lift_rev > (lift_cost * 3):
        st.success(f"**PROCEED:** Incremental Profit: **₹{int(lift_rev - lift_cost):,}**.")
    else:
        st.warning("**CAUTION:** Marginal ROI. Consider free channels.")
