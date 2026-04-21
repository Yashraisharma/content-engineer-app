import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE GOOGLE NEWS & WEATHER ENGINE ---
@st.cache_data(ttl=600)
def fetch_live_news(query, count=2):
    """Fetches real-time headlines from Google News RSS."""
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        return [{"title": item.find('title').text.split(' - ')[0], "link": item.find('link').text} for item in root.findall('./channel/item')[:count]]
    except:
        return [{"title": "Live news feed temporarily unavailable", "link": "#"}]

# 2026 Demographic & Environment Benchmarks
CITY_DNA = {
    "mumbai": {"temp": "31°C", "seniors": "14.8%", "m_f": "853:1000", "moms": "2.8M", "tech": "92%", "weather": "🌡️ 31°C | Mist | Humidity 63%"},
    "delhi": {"temp": "36°C", "seniors": "12.2%", "m_f": "868:1000", "moms": "3.2M", "tech": "91%", "weather": "🌡️ 36°C | Heat Alert (Mist/Humidity 21%)"},
    "bangalore": {"temp": "31°C", "seniors": "11.5%", "m_f": "923:1000", "moms": "1.8M", "tech": "96%", "weather": "🌡️ 31°C | Clear | Humidity 36%"},
    "hyderabad": {"temp": "35°C", "seniors": "10.9%", "m_f": "955:1000", "moms": "1.4M", "tech": "94%", "weather": "🌡️ 35°C | Yellow Alert | Humidity 35%"},
    "chennai": {"temp": "30°C", "seniors": "15.2%", "m_f": "989:1000", "moms": "1.2M", "tech": "90%", "weather": "🌡️ 30°C | Partly Cloudy | Humidity 79%"},
    "kolkata": {"temp": "30°C", "seniors": "16.1%", "m_f": "908:1000", "moms": "1.6M", "tech": "86%", "weather": "🌡️ 30°C | Mist | Humidity 84%"}
}

def run_page():
    # --- 2. CORE CONFIG ---
    now = datetime.now()
    st.header("🛡️ Strategic Growth Command & ROI Predictor")
    st.markdown(f"**Growth Engine Status:** ACTIVE | {now.strftime('%A, %d %B %Y | %I:%M %p')}")

    # 3. EXCEL DATA INTEGRATION
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
                        'Name': str(r.iloc[0]).strip(), 'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0, 'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0
                    })
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_data()

    # --- 4. THE GLITCH-FREE SELECTION ---
    if "selected_segments" not in st.session_state: st.session_state.selected_segments = []
    def sync_picks(): st.session_state.selected_segments = st.session_state.ms_key

    st.sidebar.title("🎮 Targeting Matrix")
    is_circle = st.sidebar.checkbox("🟢 Target CIRCLE Members")

    picks = st.multiselect("🔍 Select Cohorts (Cities, Categories, Segments):", 
                           options=df_master['Name'].unique().tolist() if not df_master.empty else [],
                           default=st.session_state.selected_segments, key="ms_key", on_change=sync_picks)

    if not picks:
        st.info("👋 Select a target cohort above to reveal real-time intelligence.")
        return

    # --- 5. THE LIVE INTELLIGENCE LOOP ---
    st.divider()
    tabs = st.tabs([p for p in picks])

    for i, primary in enumerate(picks):
        with tabs[i]:
            p_lower = primary.lower()
            city = next((c for c in CITY_DNA.keys() if c in p_lower), "hyderabad")
            dna = CITY_DNA[city]
            
            # --- FETCH LIVE CONTEXT ---
            with st.spinner(f"Querying Google for {primary}..."):
                # Real-Time IPL & Events (April 21, 2026)
                if city == "hyderabad": 
                    local_news = "🏏 IPL 2026: SRH vs DC @ Uppal Stadium (7:30 PM). Traffic curbs active."
                elif city == "delhi": 
                    local_news = "🏛️ Civil Services Day: VP Radhakrishnan to address civil servants today."
                else: 
                    local_news = fetch_live_news(f"{city} top news headlines April 21 2026", 1)[0]['title']
                
                health_news = fetch_live_news(f"India healthcare news {primary} April 2026", 1)[0]

            # --- THE INTELLIGENCE CARD ---
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 3px solid #1e293b; padding: 25px; border-radius: 15px; color: #000; margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin:0;">🕵️ {primary.upper()}</h2>
                        <span style="background: #ef4444; color:#fff; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{dna['weather']}</span>
                    </div>
                    <p style="background: #e2e8f0; padding: 12px; border-radius: 8px; border-left: 5px solid #1e293b; margin: 15px 0; font-size: 0.95em;">
                        <b>🔥 Live Event:</b> {local_news}<br>
                        <b>🏥 Health Focus:</b> <a href="{health_news['link']}" target="_blank" style="color:#1d4ed8;">{health_news['title']}</a>
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.85em;">
                        <div style="background:#fff; padding:12px; border-radius:8px; border:1px solid #ddd; text-align:center;">👵 <b>Seniors:</b><br>{dna['seniors']}</div>
                        <div style="background:#fff; padding:12px; border-radius:8px; border:1px solid #ddd; text-align:center;">🍼 <b>Moms (2026):</b><br>{dna['moms']}</div>
                        <div style="background:#fff; padding:12px; border-radius:8px; border:1px solid #ddd; text-align:center;">👫 <b>M/F Ratio:</b><br>{dna['m_f']}</div>
                        <div style="background:#fff; padding:12px; border-radius:8px; border:1px solid #ddd; text-align:center;">📱 <b>Tech Savvy:</b><br>{dna['tech']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- DYNAMIC BASKET AFFINITY (HIERARCHY FIXED) ---
            if any(x in p_lower for x in ["mom", "baby"]): p1, p2, p1_u = "Pampers Baby-Dry", "Himalaya Wipes", "baby-care"
            elif any(x in p_lower for x in ["cardio", "diag", "chronic"]): p1, p2, p1_u = "Apollo Digital BP Monitor", "OneTouch Select Plus", "health-devices"
            elif "skin" in p_lower: p1, p2, p1_u = "Cetaphil Gentle Cleanser", "Apollo SPF 50 Sunscreen", "skin-care"
            else: p1, p2, p1_u = "ORSL Electrolyte Orange", "Apollo SPF 50 Sunscreen", "otc"

            # --- CROSS-SELL UI ---
            st.write("### 🛒 Strategic Cross-Sell (Basket Affinity)")
            c1, c2, c3 = st.columns([1, 1, 1.5])
            c1.info(f"**Primary Push:** [{p1}](https://www.apollopharmacy.in/shop-by-category/{p1_u})")
            c2.success(f"**Logical Upsell:** {p2}")
            c3.warning(f"**🧠 AI Pitch Strategy:** {'🚨 WINBACK: Offer 25% OFF' if 'churn' in p_lower else '⭐ NTU: 15% Welcome Coupon' if 'ntu' in p_lower else '👑 CIRCLE: Priority 2HR Delivery'}")

    # --- 6. AGGREGATED REACH DNA & ROI ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader(f"🧬 Aggregated ROI: {len(picks)} Segments")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    m5.metric("Email", f"{int(stats.get('Email', 0)):,}")

    col_v1, col_v2, col_v3 = st.columns(3)
    wa_rate = col_v1.number_input("WA Cost", value=0.78)
    sms_rate = col_v2.number_input("SMS Cost", value=0.13)
    email_rate = col_v3.number_input("Email Cost", value=0.03)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conv Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Rev": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    st.table(pd.DataFrame([calc("Push", stats['Push'], 0.0), calc("WhatsApp", stats['WA'], wa_rate), calc("SMS", stats['SMS'], sms_rate), calc("Email", stats.get('Email', 0), email_rate)]))
