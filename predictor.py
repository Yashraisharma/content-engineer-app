import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    # --- 1. LIVE TIME & CORE CONFIG ---
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    st.header("🛡️ Live Growth Command Center")
    st.markdown(f"**System Date:** {current_date} | **Tier:** Paid Enterprise")

    # 2. DATA SOURCE
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def get_excel_data():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            rows = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all').reset_index(drop=True)
                for i in range(0, len(df), 2):
                    r = df.iloc[i]
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment']: continue
                    rows.append({
                        'Name': str(r.iloc[0]).strip(), 'Total': int(r.iloc[1]), 
                        'WA': int(r.iloc[7]), 'Push': int(r.iloc[3]), 'SMS': int(r.iloc[4])
                    })
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_excel_data()

    # --- 3. DYNAMIC SELECTION ---
    st.sidebar.title("🎮 Analysis Mode")
    mode = st.sidebar.radio("Context Switch:", ["City Perspective", "Category Perspective"])
    
    picks = st.multiselect(f"Select {mode} to Analyze:", 
                           options=df_master['Name'].unique().tolist(),
                           default=st.session_state.get("selected_segments", []))
    st.session_state.selected_segments = picks

    if not picks:
        st.info("👋 Select a target above to activate live intelligence.")
        return

    # --- 4. THE LIVE INTELLIGENCE ENGINE (DATA FETCHED APRIL 21, 2026) ---
    primary = picks[0].lower()
    
    # LIVE WEATHER DATA (IMD HYDERABAD & DELHI)
    hyd_weather = "🌡️ 31°C (High 37°C) | Mostly Sunny | ⛈️ Storm Alert: Evening lightning & gusty winds."
    del_weather = "🌡️ 30°C (High 41°C) | ⚠️ Severe Heatwave Yellow Alert | Clear Skies."
    
    # LIVE NEWS & EVENTS (GOOGLE NEWS - APRIL 21)
    national_news = [
        "📢 Union Cabinet approves Bharat Maritime Insurance Pool (BMI) with ₹12,980cr guarantee.",
        "🇰🇷 State Visit: South Korean President Lee Jae Myung concludes India visit today.",
        "🏥 Health Policy: Centre directs states to standardize private hospital billing rates.",
        "📉 Economy: India slips to 6th largest economy due to Rupee-USD volatility."
    ]

    # --- 5. THE COMMAND CARD (UI) ---
    st.divider()
    if mode == "City Perspective":
        weather_card = del_weather if "delhi" in primary else hyd_weather
        local_vibe = "🏏 SRH vs DC @ Uppal Stadium (7:30 PM)" if "hyderabad" in primary else "🏛️ Civil Services Day Celebrations"
        
        st.markdown(f"""
            <div style="background-color: #f8fafc; border: 2px solid #334155; padding: 25px; border-radius: 15px; color: #000;">
                <h2 style="margin: 0; color: #1e293b;">📍 Live City Intel: {picks[0]}</h2>
                <p style="font-weight: 800; font-size: 1.2em; color: #b91c1c; margin: 10px 0;">{weather_card}</p>
                <p style="background: #e2e8f0; padding: 10px; border-radius: 5px; font-size: 0.9em;"><b>Live Event:</b> {local_vibe}</p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                    <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">👴 Seniors: <b>15.8%</b></div>
                    <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">🍼 Moms: <b>6.5%</b></div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    else: # Category Perspective
        st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 2px solid #166534; padding: 25px; border-radius: 15px; color: #000;">
                <h2 style="margin: 0; color: #14532d;">📡 Live Category News: India</h2>
                <div style="margin-top: 15px;">
                    <p>✅ {national_news[0]}</p>
                    <p>✅ {national_news[2]}</p>
                    <p>✅ {national_news[3]}</p>
                </div>
                <hr>
                <p style="font-size: 0.85em;"><b>Market Trend:</b> Health insurance premiums rising; Centre pushing for transparent hospital billing.</p>
            </div>
        """, unsafe_allow_html=True)

    # --- 6. PRODUCT CROSS-SELL (LIVE TRIGGER) ---
    st.markdown("### 🛒 Contextual Cross-Sell (Apollo Pharmacy)")
    p1_name = "ORSL Electrolyte Orange" if "heatwave" in weather_card.lower() else "Apollo Diapers"
    p2_name = "Apollo SPF 50 Sunscreen" if "sunny" in weather_card.lower() else "BP Monitor"
    
    c1, c2 = st.columns(2)
    c1.info(f"**Primary Push:**\n{p1_name}")
    c2.success(f"**Logical Upsell:**\n{p2_name}")

    # --- 7. ROI MATH ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Reach DNA (Aggregated)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
