import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE GOOGLE NEWS API (RESTORED & FIXED) ---
@st.cache_data(ttl=300)
def fetch_news(query, count=2):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = root.findall('./channel/item')
        return [{"title": i.find('title').text.split(' - ')[0], "link": i.find('link').text} for i in items[:count]]
    except:
        return [{"title": "Live news feed temporarily unavailable", "link": "#"}]

# --- 2. 2026 DEMOGRAPHIC DNA (STRICT PERCENTAGES) ---
CITY_DNA = {
    "mumbai": {"temp": "31°C", "seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%"},
    "delhi": {"temp": "36°C", "seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%"},
    "bangalore": {"temp": "31°C", "seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%"},
    "hyderabad": {"temp": "35°C", "seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%"},
    "chennai": {"temp": "30°C", "seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%"},
    "kolkata": {"temp": "30°C", "seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%"}
}

def run_page():
    now = datetime.now()
    st.header("🛡️ Strategic Growth Predictor")
    st.markdown(f"**Live Sync:** {now.strftime('%A, %d %B %Y')} | **Engine:** Live API v4")

    # --- DATA ENGINE ---
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
                        'Name': str(r.iloc[0]).strip(), 'Total': int(r.iloc[1]), 
                        'WA': int(r.iloc[7]), 'Push': int(r.iloc[3]), 'SMS': int(r.iloc[4]), 'Email': int(r.iloc[5])
                    })
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_data()

    # --- SELECTION ---
    picks = st.multiselect("🔍 Select Cohorts:", options=df_master['Name'].unique().tolist() if not df_master.empty else [], 
                           default=st.session_state.get("selected_segments", []), key="ms_key")
    st.session_state.selected_segments = picks

    if not picks:
        st.info("👋 Select cohorts above to activate live data engine.")
        return

    # --- 3. THE LIVE INTELLIGENCE ENGINE ---
    st.divider()
    tabs = st.tabs([p for p in picks])

    for i, primary in enumerate(picks):
        with tabs[i]:
            p_lower = primary.lower()
            city_key = next((c for c in CITY_DNA.keys() if c in p_lower), "hyderabad")
            dna = CITY_DNA[city_key]
            
            with st.spinner(f"Querying Google for {primary}..."):
                common_news = fetch_news(f"{city_key} top headlines April 2026", 1)[0]
                health_news = fetch_news(f"{primary} healthcare trends India April 2026", 1)[0]

            # --- INTELLIGENCE CARD ---
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 2px solid #1e293b; padding: 25px; border-radius: 15px; color: #000; margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="margin:0;">🕵️ {primary.upper()}</h2>
                        <span style="background: #ef4444; color:#fff; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{dna['temp']} | {city_key.upper()}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
                        <div style="background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd;">
                            <b>📰 1. Common News:</b><br><a href="{common_news['link']}" target="_blank" style="color:#1d4ed8;">{common_news['title']}</a>
                        </div>
                        <div style="background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd;">
                            <b>🏥 2. Health News:</b><br><a href="{health_news['link']}" target="_blank" style="color:#047857;">{health_news['title']}</a>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">👵 <b>Seniors</b><br>{dna['seniors']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">🍼 <b>Moms</b><br>{dna['moms']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">👩 <b>Females</b><br>{dna['females']}</div>
                        <div style="background:#f1f5f9; padding:12px; border-radius:8px; text-align:center;">📱 <b>Tech</b><br>{dna['tech']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- 4. EXPANDED CROSS-SELL (5 POSSIBILITIES) ---
            st.subheader("🛒 Strategic Cross-Sell: 5 Targeted Options")
            
            if "mom" in p_lower or "baby" in p_lower:
                opts = [
                    ("Pampers Baby-Dry Diapers", "baby-care/diapers", "Daily essential restock for infants."),
                    ("Himalaya Gentle Baby Wipes", "baby-care/baby-wipes", "High-affinity item for diaper basket."),
                    ("Nestle NAN PRO 2", "baby-care", "Nutritional base for growing babies."),
                    ("Apollo Baby Lotion", "apollo-personal-care", "Skincare upsell for urban moms."),
                    ("Sebamed Baby Wash", "baby-care", "Premium upsell for skin-sensitive cohorts.")
                ]
            elif "cardio" in p_lower or "diab" in p_lower:
                opts = [
                    ("Apollo BP Monitor", "health-devices", "Critical monitoring for cardio patients."),
                    ("OneTouch Select Plus Strips", "diabetes-care", "Razor-blade model for recurring strips."),
                    ("Protinex Diabetes Care", "diabetes-supplements", "Nutritional filler for chronic patients."),
                    ("Apollo Sugar-Free Protein", "diabetes-care", "Healthy supplement upsell."),
                    ("Accu-Chek Active Monitor", "health-devices", "Brand-loyal alternative upsell.")
                ]
            elif "skin" in p_lower:
                opts = [
                    ("Cetaphil Cleanser", "skin-care", "Dermatologist choice; high retention."),
                    ("Apollo SPF 50 Sunscreen", "apollo-personal-care", "Immediate need for current heatwave."),
                    ("Novology Acne Serum", "skin-care", "High-margin clinical serum upsell."),
                    ("Bioderma Sensibio H2O", "skin-care", "Premium brand upsell for urbanites."),
                    ("Apollo Aloe Vera Gel", "otc", "Post-sun exposure soothing cross-sell.")
                ]
            else:
                opts = [
                    ("ORSL Electrolyte Orange", "otc", "Seasonal hydration due to current temp."),
                    ("Apollo Multivitamins", "vitamins-and-supplements", "Daily wellness anchor."),
                    ("Seven Seas Cod Liver Oil", "elderly-care", "Immunity booster for senior segments."),
                    ("Dabur Honey (Immunity)", "health-drinks", "Natural health cross-sell."),
                    ("Apollo Digital Thermometer", "health-devices", "Household essential for general cohorts.")
                ]

            cols = st.columns(5)
            for idx, (name, link, reason) in enumerate(opts):
                with cols[idx]:
                    st.info(f"**#{idx+1}:** {name}")
                    st.write(f"*{reason}*")
                    st.markdown(f"[🔗 Shop Now](https://www.apollopharmacy.in/shop-by-category/{link})")

    # --- 5. ROI FORECAST ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Aggregated Reach & ROI")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", f"{int(stats['Total']):,}"); m2.metric("WA", f"{int(stats['WA']):,}"); m3.metric("Push", f"{int(stats['Push']):,}"); m4.metric("SMS", f"{int(stats['SMS']):,}"); m5.metric("Email", f"{int(stats['Email']):,}")
