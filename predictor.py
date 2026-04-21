import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE GOOGLE NEWS API ---
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
    st.header("🛡️ Strategic Growth Predictor")
    st.markdown(f"**Live Sync:** {now.strftime('%A, %d %B %Y')} | **Engine:** Live API v5")

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
    if "selected_segments" not in st.session_state: st.session_state.selected_segments = []
    def sync_picks(): st.session_state.selected_segments = st.session_state.ms_key

    picks = st.multiselect("🔍 Select Cohorts:", options=df_master['Name'].unique().tolist() if not df_master.empty else [], 
                           default=st.session_state.selected_segments, key="ms_key", on_change=sync_picks)

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

            # --- 4. THE 5-ROW CROSS-SELL TABLE (WITH DIRECT APOLLO LINKS) ---
            st.subheader("🛒 Strategic Cross-Sell Matrix")
            
            # Helper function to generate Markdown links
            def apl_link(name, path):
                return f'<a href="https://www.apollopharmacy.in/shop-by-category/{path}" target="_blank">{name}</a>'

            if "mom" in p_lower or "baby" in p_lower:
                xsell_data = [
                    ["Pampers / Huggies Diapers", apl_link("Himalaya / Littles Baby Wipes", "baby-care/baby-wipes"), "90% basket affinity; wipes are consumed per diaper change."],
                    ["Baby Formula (NAN/Similac)", apl_link("Bottle Sterilizer / Liquid Cleanser", "baby-care"), "Safety protocol intent; requires sterile feeding gear."],
                    ["Baby Body Wash / Soap", apl_link("Baby Lotion / Massage Oil", "baby-care"), "Complete post-bath routine bundling."],
                    ["Diaper Rash Cream", apl_link("Baby Powder (Talc-free)", "baby-care"), "Complete moisture and rash prevention protocol."],
                    ["Teethers / Pacifiers", apl_link("Colic Drops / Gripe Water", "baby-care"), "Symptom pairing; teething often causes gastric distress."]
                ]
            elif "cardio" in p_lower or "diab" in p_lower or "diag" in p_lower:
                xsell_data = [
                    ["Glucometer Strips", apl_link("Lancets & Alcohol Swabs", "diabetes-care"), "Razor-blade model; essential consumables for blood testing."],
                    ["Blood Pressure Monitor", apl_link("Digital Thermometer / Pulse Ox", "health-devices"), "Baseline home-health kit completion for chronic patients."],
                    ["Heart Statins / Meds", apl_link("Omega-3 / Fish Oil Capsules", "elderly-care"), "Supplementing prescription care with heart-healthy lipids."],
                    ["Diabetic Footwear / Socks", apl_link("Diabetic Foot Care Cream", "diabetes-care"), "Neuropathy prevention and daily care protocol."],
                    ["Artificial Sweeteners", apl_link("Sugar-Free Protein Powder", "diabetes-supplements"), "Lifestyle transition; moving to a complete diabetic diet."]
                ]
            elif "skin" in p_lower:
                xsell_data = [
                    ["Acne Face Wash / Salicylic", apl_link("Non-Comedogenic Sunscreen", "apollo-personal-care"), "Core daytime routine; prevents post-acne hyperpigmentation."],
                    ["AHA / BHA Exfoliant Serum", apl_link("Ceramide Heavy Moisturizer", "skin-care"), "Barrier repair necessity after chemical exfoliation."],
                    ["Vitamin C Serum", apl_link("SPF 50 Sunscreen", "apollo-personal-care/sun-care"), "Vitamin C boosts SPF efficacy and prevents photo-oxidation."],
                    ["Body Wash / Shower Gel", apl_link("Loofah / Body Exfoliator", "personal-care"), "Bath utility and routine completion bundling."],
                    ["Aloe Vera Gel", apl_link("Calamine Lotion / Lacto Calamine", "skin-care"), "Summer heatwave soothing bundle for irritated skin."]
                ]
            else: # General / Urban
                xsell_data = [
                    ["ORSL / Electrolytes", apl_link("SPF 50 Sunscreen / Odomos", "apollo-personal-care"), "Complete outdoor heatwave and vector protection kit."],
                    ["Multivitamins (Daily)", apl_link("Omega-3 Supplements", "vitamins-and-supplements"), "Premium daily wellness stack for urban professionals."],
                    ["Antacids / Digene / Eno", apl_link("Probiotic Supplements", "otc"), "Gut health restoration and flora balance after acute acidity."],
                    ["Paracetamol / Dolo 650", apl_link("Vicks / Cough Drops / Honitus", "otc"), "Broad-spectrum viral symptom coverage (Fever + Throat)."],
                    ["Anti-Hairfall Shampoo", apl_link("Hair Vitalizer Serum / Oil", "personal-care"), "Comprehensive scalp care routine for urban stress/water issues."]
                ]

            df_xsell = pd.DataFrame(xsell_data, columns=["User Purchase", "Push", "Reason why suggested"])
            
            # Using st.markdown to render the HTML links inside the dataframe correctly
            st.markdown(df_xsell.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("") # Adds a little padding after the table

    # --- 5. ROI FORECAST ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Aggregated Reach & ROI")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", f"{int(stats['Total']):,}"); m2.metric("WA", f"{int(stats['WA']):,}"); m3.metric("Push", f"{int(stats['Push']):,}"); m4.metric("SMS", f"{int(stats['SMS']):,}"); m5.metric("Email", f"{int(stats['Email']):,}")

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
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"}

    st.table(pd.DataFrame([
        calc("Mobile Push", stats['Push'], 0.0), calc("WhatsApp", stats['WA'], wa_rate), 
        calc("SMS", stats['SMS'], sms_rate), calc("Email", stats['Email'], email_rate)
    ]))
