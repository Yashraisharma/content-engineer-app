import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    # --- 1. CORE CONFIG & LIVE TIME ---
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    st.header("🛡️ Live Growth Command Center")
    st.markdown(f"**System Date:** {current_date} | **Tier:** Paid Enterprise")

    # --- 2. DATA SOURCE ---
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
        except Exception as e: 
            st.error(f"Excel Load Error: {e}")
            return pd.DataFrame()

    df_master = get_excel_data()

    # --- 3. DYNAMIC SELECTION ---
    st.sidebar.title("🎮 Analysis Mode")
    mode = st.sidebar.radio("Context Switch:", ["City Perspective", "Category Perspective"])
    
    # Glitch-free memory
    if "selected_segments" not in st.session_state:
        st.session_state.selected_segments = []

    def sync_picks():
        st.session_state.selected_segments = st.session_state.ms_key

    picks = st.multiselect(
        f"Select {mode} to Analyze:", 
        options=df_master['Name'].unique().tolist() if not df_master.empty else [],
        default=st.session_state.selected_segments,
        key="ms_key",
        on_change=sync_picks
    )

    if not picks:
        st.info("👋 Select a target above to activate live intelligence.")
        return

    # --- 4. THE LIVE INTELLIGENCE ENGINE ---
    primary = picks[0].lower()
    
    # Simulated Live Data Feeds
    hyd_weather = "🌡️ 31°C | Mostly Sunny | ⛈️ Storm Alert: Evening lightning & gusty winds."
    del_weather = "🌡️ 30°C (High 41°C) | ⚠️ Severe Heatwave Yellow Alert."
    national_news = [
        "📢 Union Cabinet approves Bharat Maritime Insurance Pool.",
        "🇰🇷 State Visit: South Korean President concludes India visit today.",
        "🏥 Health Policy: Centre directs states to standardize private hospital billing."
    ]

    # --- 5. THE COMMAND CARD (UI) ---
    st.divider()
    if mode == "City Perspective":
        # City Logic
        is_delhi = "delhi" in primary
        weather_card = del_weather if is_delhi else hyd_weather
        local_vibe = "🏛️ Civil Services Day Celebrations (Traffic curbs)" if is_delhi else "🏏 SRH vs DC @ Uppal Stadium (7:30 PM)"
        
        st.markdown(f"""
            <div style="background-color: #f8fafc; border: 2px solid #334155; padding: 25px; border-radius: 15px; color: #000;">
                <h2 style="margin: 0; color: #1e293b;">📍 Live City Intel: {picks[0]}</h2>
                <p style="font-weight: 800; font-size: 1.2em; color: #b91c1c; margin: 10px 0;">{weather_card}</p>
                <p style="background: #e2e8f0; padding: 10px; border-radius: 5px; font-size: 0.9em;"><b>Live Event:</b> {local_vibe}</p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 20px;">
                    <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">👵 Seniors: <b>15.8%</b></div>
                    <div style="background: white; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #ddd;">🍼 Moms: <b>6.5%</b></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # BASKET AFFINITY LOGIC (CITY/WEATHER)
        if "heatwave" in weather_card.lower():
            p1, p2 = "ORSL Electrolyte Orange", "Apollo SPF 50 Sunscreen"
        elif "storm" in weather_card.lower() or "rain" in weather_card.lower():
            p1, p2 = "Apollo Pharmacy First Aid Kit", "Odomos Mosquito Repellent"
        else:
            p1, p2 = "Apollo Life Multivitamins", "Dabur Honey (Immunity)"

    else: 
        # Category Logic
        st.markdown(f"""
            <div style="background-color: #f0fdf4; border: 2px solid #166534; padding: 25px; border-radius: 15px; color: #000;">
                <h2 style="margin: 0; color: #14532d;">📡 Live Category News: India</h2>
                <div style="margin-top: 15px;">
                    <p>✅ {national_news[0]}</p>
                    <p>✅ {national_news[1]}</p>
                    <p>✅ {national_news[2]}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # BASKET AFFINITY LOGIC (CATEGORY/DEMOGRAPHIC)
        if any(x in primary for x in ["mom", "baby", "pediatric"]):
            p1, p2 = "Pampers Baby-Dry Diapers", "Himalaya Gentle Baby Wipes"
        elif any(x in primary for x in ["cardio", "diab", "chronic", "senior"]):
            p1, p2 = "Apollo Pharmacy Digital BP Monitor", "Apollo Life Sugar-Free Protein"
        else:
            p1, p2 = "Sensodyne Whitening Toothpaste", "Listerine Mouthwash"

    # --- 6. PRODUCT CROSS-SELL UI ---
    st.markdown("### 🛒 Contextual Cross-Sell (Basket Affinity)")
    c1, c2 = st.columns(2)
    c1.info(f"**Primary Campaign Push:**\n{p1}")
    c2.success(f"**Logical Upsell Match:**\n{p2}")

    # --- 7. REACH DNA & ROI MATH ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Reach DNA (Aggregated)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    
    st.divider()
    st.subheader("🔮 Campaign ROI Forecast")
    
    col_v1, col_v2 = st.columns(2)
    wa_rate = col_v1.number_input("WA Cost (Karix)", value=0.78)
    sms_rate = col_v2.number_input("SMS Cost (Vi)", value=0.13)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc_channel(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Rev": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    table = [
        calc_channel("Mobile Push", stats['Push'], 0.0),
        calc_channel("WhatsApp", stats['WA'], wa_rate),
        calc_channel("SMS", stats['SMS'], sms_rate)
    ]
    st.table(pd.DataFrame(table))
