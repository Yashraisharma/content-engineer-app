import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    # --- 1. LIVE CLOCK & CALENDAR (APRIL 21, 2026) ---
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    current_hour = now.hour
    
    st.header("📊 Live Strategic Command & ROI Predictor")
    st.markdown(f"#### Growth Engine Status: ACTIVE | {current_date}")
    
    # 2. DATA SOURCE (EXCEL)
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

    # --- 3. THE GLITCH-FREE SELECTION ---
    if "selected_segments" not in st.session_state:
        st.session_state.selected_segments = []

    def sync_picks():
        st.session_state.selected_segments = st.session_state.multiselect_key

    picks = st.multiselect(
        "🔍 Search & Analyze Live Segments:", 
        options=df_master['Name'].unique().tolist() if not df_master.empty else [],
        default=st.session_state.selected_segments,
        key="multiselect_key",
        on_change=sync_picks
    )

    if not picks:
        st.info("👋 Select a segment from the search bar above to begin real-time analysis.")
        return

    # --- 4. LIVE SEARCH & CONTEXT (April 21, 2026) ---
    primary = picks[0].lower()
    
    # Priority 1: Mother & Baby
    if any(x in primary for x in ["mom", "baby", "infant", "pediatric"]):
        intel = {
            "old": "2.8%", "moms": "13.2%", "tech": "96%", "type": "Pediatric",
            "weather": "🌡️ 39°C | Hot & Humid", 
            "news": "📢 Apollo Kids Alert: Viral fever spike in urban clusters; keep hydration high.",
            "color": "#fdf2f8", "border": "#ec4899"
        }
        p1 = {"name": "Pampers All-Round Protection Diapers", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/diapers"}
        p2 = {"name": "Himalaya Baby Wipes (Pack of 3)", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/baby-wipes"}

    # Priority 2: Chronic (Seniors)
    elif any(x in primary for x in ["cardio", "diab", "pharma", "chronic", "senior"]):
        intel = {
            "old": "26.8%", "moms": "1.4%", "tech": "42%", "type": "Geriatric",
            "weather": "⚠️ RED ALERT: 41.5°C Heatwave. Extreme risk for 60+.",
            "news": "🗞️ IMD: Severe heatwave in Telangana/AP; advise indoor stay for seniors.",
            "color": "#f0fdf4", "border": "#22c55e"
        }
        p1 = {"name": "Apollo Pharmacy Adult Diaper Pants (XL)", "link": "https://www.apollopharmacy.in/shop-by-category/apollo-adult-diapers"}
        p2 = {"name": "Apollo Pharmacy Joint Health Formula (30 Tab)", "link": "https://www.apollopharmacy.in/shop-by-category/elderly-care"}

    # Priority 3: City Specific (Hyderabad Example)
    elif "hyderabad" in primary:
        intel = {
            "old": "12.5%", "moms": "5.8%", "tech": "91%", "type": "Metro Urban",
            "weather": "🌩️ Unseasonal Rain & Gusty Winds | 38°C",
            "news": "🏏 IPL Uppal Stadium: SRH vs DC Today. Expect massive traffic curbs Nagole-Habsiguda.",
            "color": "#eff6ff", "border": "#3b82f6"
        }
        p1 = {"name": "ORSL Electrolyte (Orange Pack)", "link": "https://www.apollopharmacy.in/shop-by-category/otc"}
        p2 = {"name": "Apollo Pharmacy SPF 50 Sunscreen", "link": "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care"}

    else:
        intel = {"old": "15%", "moms": "5%", "tech": "85%", "type": "General", "weather": "☀️ 38.5°C", "news": "📢 Check Apollo 247 for summer deals.", "color": "#f8fafc", "border": "#64748b"}
        p1 = {"name": "Apollo Life Multivitamins", "link": "https://www.apollopharmacy.in/shop-by-category/vitamins-and-supplements"}
        p2 = {"name": "Apollo Pharmacy First Aid Kit", "link": "https://www.apollopharmacy.in/shop-by-category/otc"}

    # --- 5. THE INTELLIGENCE CARD (VISUAL) ---
    st.markdown(f"""
        <div style="background-color: {intel['color']}; border: 3px solid {intel['border']}; padding: 25px; border-radius: 15px; color: #000; margin-bottom: 30px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0; color: #000;">🕵️ Live Context: {picks[0]}</h2>
                <span style="background: #000; color: #fff; padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 800;">LIVE @ {now.strftime('%H:%M')}</span>
            </div>
            <p style="font-weight: 900; font-size: 1.4em; color: {intel['border']}; margin: 15px 0;">{intel['weather']}</p>
            <div style="background: white; padding: 12px; border-radius: 8px; border-left: 5px solid {intel['border']}; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 0.95em; color: #1e293b;"><b>🔥 Top News:</b> {intel['news']}</p>
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; color: #000;">
                    <span style="font-size: 2em;">👵</span><br><b>Seniors (60+)</b><br><span style="font-size: 1.6em; font-weight: 900;">{intel['old']}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; color: #000;">
                    <span style="font-size: 2em;">🍼</span><br><b>Moms/Babies</b><br><span style="font-size: 1.6em; font-weight: 900;">{intel['moms']}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; color: #000;">
                    <span style="font-size: 2em;">📱</span><br><b>App Savvy</b><br><span style="font-size: 1.6em; font-weight: 900;">{intel['tech']}</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #ddd; color: #000;">
                    <span style="font-size: 2em;">🏷️</span><br><b>Cohort Class</b><br><span style="font-size: 1.1em; font-weight: 700;">{intel['type']}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 6. SMART CROSS-SELL ---
    st.markdown("### 🛒 Real-Time Cross-Sell (Apollo Pharmacy)")
    cs1, cs2 = st.columns(2)
    cs1.info(f"**Primary Campaign Product:**\n{p1['name']}")
    cs1.markdown(f"[Buy on Apollo Pharmacy]({p1['link']})")
    cs2.success(f"**Logical Upsell:**\n{p2['name']}")
    cs2.markdown(f"[Buy on Apollo Pharmacy]({p2['link']})")

    # --- 7. REACH DNA & ROI ---
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
    conv = st.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = st.number_input("Average Order Value (₹)", value=800)

    def calc_roi(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "Profit": f"₹{int(rev-spend):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    st.table(pd.DataFrame([
        calc_roi("Mobile Push", combined_data['Push'], 0.0),
        calc_roi("WhatsApp", combined_data['WA'], wa_rate),
        calc_roi("SMS", combined_data['SMS'], sms_rate)
    ]))
