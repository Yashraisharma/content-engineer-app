import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE GOOGLE NEWS ENGINE ---
@st.cache_data(ttl=600)
def fetch_live_news(query, count=2):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        return [{"title": item.find('title').text.split(' - ')[0], "link": item.find('link').text} for item in root.findall('./channel/item')[:count]]
    except:
        return [{"title": "Live news feed temporarily unavailable", "link": "#"}]

# --- 2. 2026 DEMOGRAPHIC DNA (PERCENTAGES) ---
CITY_DNA = {
    "mumbai": {"temp": "31°C", "seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "weather": "🌡️ 31°C | Mist | Humidity 63%"},
    "delhi": {"temp": "36°C", "seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "weather": "🌡️ 36°C | Heat Alert | Humidity 21%"},
    "bangalore": {"temp": "31°C", "seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "weather": "🌡️ 31°C | Clear | Humidity 36%"},
    "hyderabad": {"temp": "35°C", "seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "weather": "🌡️ 35°C | Yellow Alert | Humidity 35%"},
    "chennai": {"temp": "30°C", "seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "weather": "🌡️ 30°C | Partly Cloudy | Humidity 79%"},
    "kolkata": {"temp": "30°C", "seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "weather": "🌡️ 30°C | Mist | Humidity 84%"}
}

def run_page():
    now = datetime.now()
    st.header("🛡️ Apollo 247: Strategic Growth Command")
    st.markdown(f"**Growth Engine Status:** ACTIVE | {now.strftime('%A, %d %B %Y | %I:%M %p')}")

    # --- 3. EXCEL DATA INTEGRATION ---
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
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment']: continue
                    rows.append({
                        'Name': str(r.iloc[0]).strip(), 
                        'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 
                        'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0, 
                        'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0
                    })
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_data()

    # --- 4. TARGET SELECTION ---
    if "selected_segments" not in st.session_state: st.session_state.selected_segments = []
    def sync_picks(): st.session_state.selected_segments = st.session_state.ms_key

    picks = st.multiselect("🔍 Select Target Cohorts (City, Category, Segment):", 
                           options=df_master['Name'].unique().tolist() if not df_master.empty else [],
                           default=st.session_state.selected_segments, key="ms_key", on_change=sync_picks)

    if not picks:
        st.info("👋 Select a target cohort to reveal live context and cross-sell options.")
        return

    # --- 5. LIVE INTELLIGENCE LOOP ---
    st.divider()
    tabs = st.tabs([p for p in picks])

    for i, primary in enumerate(picks):
        with tabs[i]:
            p_lower = primary.lower()
            city_key = next((c for c in CITY_DNA.keys() if c in p_lower), "hyderabad")
            dna = CITY_DNA[city_key]
            
            with st.spinner(f"Querying Google for {primary}..."):
                news_feed = fetch_live_news(f"{city_key} {primary} health update April 2026", 1)[0]

            # --- INTELLIGENCE CARD ---
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 2px solid #1e293b; padding: 25px; border-radius: 15px; color: #000; margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin:0;">🕵️ {primary.upper()}</h2>
                        <span style="background: #ef4444; color:#fff; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{dna['weather']}</span>
                    </div>
                    <p style="background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd; margin: 15px 0;">
                        <b>🏥 Live Health Context:</b> <a href="{news_feed['link']}" target="_blank" style="color:#1d4ed8;">{news_feed['title']}</a>
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">👵 <b>Seniors</b><br>{dna['seniors']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">🍼 <b>Moms</b><br>{dna['moms']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">👩 <b>Females</b><br>{dna['females']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">📱 <b>Tech Savvy</b><br>{dna['tech']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- 6. EXPANDED CROSS-SELL WITH REASONS & LINKS ---
            st.subheader("🛒 Strategic Cross-Sell Varieties")
            
            if "mom" in p_lower or "baby" in p_lower:
                options = [
                    {"p": "Pampers Baby-Dry Diapers", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/diapers", "reason": "Anchor item; high-frequency restock needed for infants."},
                    {"p": "Himalaya Gentle Baby Wipes", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care/baby-wipes", "reason": "High basket affinity with diapers; bulky essential."},
                    {"p": "Nestlé NAN PRO 2", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care", "reason": "Targeted nutrition for stage-2 infants; high trust brand."}
                ]
            elif "cardio" in p_lower or "diab" in p_lower or "diag" in p_lower:
                options = [
                    {"p": "Apollo Digital BP Monitor", "link": "https://www.apollopharmacy.in/shop-by-category/health-devices", "reason": "Essential for cardio tracking; high AOV device."},
                    {"p": "OneTouch Select Plus Strips", "link": "https://www.apollopharmacy.in/shop-by-category/diabetes-care", "reason": "Razor-blade model: high recurring purchase intent."},
                    {"p": "Apollo Life Sugar-Free Protein", "link": "https://www.apollopharmacy.in/shop-by-category/diabetes-supplements", "reason": "Nutritional management for diabetic patients."}
                ]
            elif "skin" in p_lower:
                options = [
                    {"p": "Cetaphil Gentle Cleanser", "link": "https://www.apollopharmacy.in/shop-by-category/skin-care", "reason": "Dermatologist choice; high loyalty in sensitive skin cohorts."},
                    {"p": "Apollo SPF 50 Sunscreen", "link": "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care", "reason": "Immediate need due to current {dna['temp']} heatwave alerts."},
                    {"p": "Novology Acne Serum", "link": "https://www.apollopharmacy.in/shop-by-category/skin-care", "reason": "Clinical up-sell for younger urban demographics."}
                ]
            else:
                options = [
                    {"p": "ORSL Electrolyte Orange", "link": "https://www.apollopharmacy.in/shop-by-category/otc", "reason": "Seasonal hydration essential for current heatwave."},
                    {"p": "Apollo Life Multivitamins", "link": "https://www.apollopharmacy.in/shop-by-category/vitamins-and-supplements", "reason": "Daily wellness anchor for urban professionals."},
                    {"p": "Seven Seas Cod Liver Oil", "link": "https://www.apollopharmacy.in/shop-by-category/elderly-care", "reason": "Immunity booster for senior-heavy segments."}
                ]

            cols = st.columns(3)
            for idx, item in enumerate(options):
                with cols[idx]:
                    st.info(f"**Option {idx+1}:**\n{item['p']}")
                    st.write(f"*{item['reason']}*")
                    st.markdown(f"[🔗 Check on Apollo Pharmacy]({item['link']})")

    # --- 7. REACH DNA & ROI MATH (ALL CHANNELS RESTORED) ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Aggregated Reach & ROI Forecast")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    m5.metric("Email", f"{int(stats.get('Email', 0)):,}") # RESTORED

    cv1, cv2, cv3 = st.columns(3)
    wa_rate = cv1.number_input("WA Cost (Karix)", value=0.78)
    sms_rate = cv2.number_input("SMS Cost (Vi)", value=0.13)
    email_rate = cv3.number_input("Email Cost (Netcore)", value=0.03) # RESTORED
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc_channel(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Rev": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    st.table(pd.DataFrame([
        calc_channel("Mobile Push", stats['Push'], 0.0),
        calc_channel("WhatsApp", stats['WA'], wa_rate),
        calc_channel("SMS", stats['SMS'], sms_rate),
        calc_channel("Email", stats.get('Email', 0), email_rate) # RESTORED
    ]))
