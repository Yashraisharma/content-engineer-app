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
                        'Name': str(r.iloc[0]).strip(), 
                        'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 
                        'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0,
                        'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0  # Restored Email Extraction
                    })
            return pd.DataFrame(rows)
        except Exception as e: 
            st.error(f"Excel Load Error: {e}")
            return pd.DataFrame()

    df_master = get_excel_data()

    # --- 3. DYNAMIC SELECTION ---
    st.sidebar.title("🎮 Analysis Mode")
    mode = st.sidebar.radio("Context Switch:", ["City Perspective", "Category Perspective"])
    
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
    
    # Simulated Live Data Feeds (April 21, 2026)
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

        if "heatwave" in weather_card.lower():
            p1, p1_url = "ORSL Electrolyte Orange", "https://www.apollopharmacy.in/shop-by-category/otc"
            p2, p2_url = "Apollo SPF 50 Sunscreen", "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care/sun-care"
        elif "storm" in weather_card.lower() or "rain" in weather_card.lower():
            p1, p1_url = "Apollo Pharmacy First Aid Kit", "https://www.apollopharmacy.in/shop-by-category/otc"
            p2, p2_url = "Odomos Mosquito Repellent", "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care"
        else:
            p1, p1_url = "Apollo Life Multivitamins", "https://www.apollopharmacy.in/shop-by-category/vitamins-and-supplements"
            p2, p2_url = "Dabur Honey (Immunity)", "https://www.apollopharmacy.in/shop-by-category/health-drinks"

    else: 
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

        if any(x in primary for x in ["mom", "baby", "pediatric"]):
            p1, p1_url = "Pampers Baby-Dry Diapers", "https://www.apollopharmacy.in/shop-by-category/baby-care/diapers"
            p2, p2_url = "Himalaya Gentle Baby Wipes", "https://www.apollopharmacy.in/shop-by-category/baby-care/baby-wipes"
        elif any(x in primary for x in ["cardio", "diab", "chronic", "senior"]):
            p1, p1_url = "Apollo Pharmacy Digital BP Monitor", "https://www.apollopharmacy.in/shop-by-category/health-devices/bp-monitors"
            p2, p2_url = "Apollo Life Sugar-Free Protein", "https://www.apollopharmacy.in/shop-by-category/diabetes-care"
        else:
            p1, p1_url = "Sensodyne Whitening Toothpaste", "https://www.apollopharmacy.in/shop-by-category/personal-care"
            p2, p2_url = "Listerine Mouthwash", "https://www.apollopharmacy.in/shop-by-category/personal-care"

    # --- 6. PRODUCT CROSS-SELL UI ---
    st.markdown("### 🛒 Contextual Cross-Sell (Basket Affinity)")
    c1, c2 = st.columns(2)
    c1.info(f"**Primary Campaign Push:**\n{p1}")
    c1.markdown(f"[🔗 Buy on Apollo Pharmacy]({p1_url})")
    c2.success(f"**Logical Upsell Match:**\n{p2}")
    c2.markdown(f"[🔗 Buy on Apollo Pharmacy]({p2_url})")

    # --- 7. REACH DNA & ROI MATH ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Reach DNA (Aggregated)")
    
    # Restored 5 Columns for Email
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    m5.metric("Email", f"{int(stats.get('Email', 0)):,}") # Restored Email
    
    st.divider()
    st.subheader("🔮 Campaign ROI Forecast")
    
    # Restored 3 Vendor Rate columns to include Email
    col_v1, col_v2, col_v3 = st.columns(3)
    wa_rate = col_v1.number_input("WA Cost (Karix)", value=0.78)
    sms_rate = col_v2.number_input("SMS Cost (Vi)", value=0.13)
    email_rate = col_v3.number_input("Email Cost (Netcore)", value=0.03) # Restored Email Cost
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc_channel(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {
            "Channel": name, 
            "Reach": f"{int(reach):,}", 
            "Spend": f"₹{int(spend):,}", 
            "Rev": f"₹{int(rev):,}", 
            "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"
        }

    # Restored Email to the ROI Table
    table = [
        calc_channel("Mobile Push", stats['Push'], 0.0),
        calc_channel("WhatsApp", stats['WA'], wa_rate),
        calc_channel("SMS", stats['SMS'], sms_rate),
        calc_channel("Email", stats.get('Email', 0), email_rate)
    ]
    st.table(pd.DataFrame(table))
