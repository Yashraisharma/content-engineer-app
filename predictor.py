import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE UTILITIES ---
@st.cache_data(ttl=300)
def fetch_news(query, count=2):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        return [{"title": i.find('title').text.split(' - ')[0], "link": i.find('link').text} for i in root.findall('./channel/item')[:count]]
    except: return [{"title": "News Feed Offline", "link": "#"}]

@st.cache_data(ttl=300)
def fetch_live_weather(city):
    try:
        # Added '&m' to force Metric units (Celsius)
        url = f"https://wttr.in/{city}?format=%t+|+%C+|+Humidity:+%h&m"
        res = requests.get(url, timeout=5)
        return f"🌡️ {res.text.strip()}" if res.status_code == 200 else "🌡️ Live Weather Syncing..."
    except: return "🌡️ Weather Service Offline"

# --- 2. DEMOGRAPHICS & SEGMENTS ---
DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%"}
}

SEGMENT_DEFS = {
    "ntu": "Non-Transacting Users (0 transactions in 60 days)",
    "churn": "Old users coming every 30 days and transacting",
    "winback": "Old NTU users coming back",
    "active": "Users with 1, 2, or 3 transactions only",
    "power": "Users hitting their 4th transaction",
    "enhancement": "High-volume users with many transactions",
    "new registered": "Newly registered users without transaction history",
    "circle": "Premium Apollo Circle Subscription members"
}

def run_page():
    now = datetime.now()
    st.set_page_config(layout="wide")
    st.header("🛡️ Strategic Growth Predictor")

    # --- 3. DATA LOAD ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    @st.cache_data
    def get_data():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            rows = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all')
                for i in range(0, len(df), 2):
                    r = df.iloc[i]
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment']: continue
                    rows.append({'Name': str(r.iloc[0]).strip(), 'Total': int(r.iloc[1]), 'WA': int(r.iloc[7]), 'Push': int(r.iloc[3]), 'SMS': int(r.iloc[4]), 'Email': int(r.iloc[5])})
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_data()
    
    if "selected_segments" not in st.session_state: st.session_state.selected_segments = []
    def sync_picks(): st.session_state.selected_segments = st.session_state.ms_key

    picks = st.multiselect("🔍 Select Target Cohorts:", options=df_master['Name'].unique().tolist() if not df_master.empty else [], default=st.session_state.selected_segments, key="ms_key", on_change=sync_picks)

    if not picks:
        st.info("👋 Select cohorts above to activate live intelligence.")
        return

    # --- 4. ENGINE TABS ---
    st.divider()
    tabs = st.tabs([p for p in picks])

    def apl_link(display_name, query):
        url = f"https://www.apollopharmacy.in/search-medicines/{query.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {display_name}</a>'

    for i, primary in enumerate(picks):
        with tabs[i]:
            p_lower = primary.lower()
            city_key = next((c for c in DEMOGRAPHICS.keys() if c in p_lower), "hyderabad")
            dna = DEMOGRAPHICS[city_key]
            
            # Segment ID
            seg_key = next((k for k in SEGMENT_DEFS.keys() if k in p_lower), "active")
            
            with st.spinner("Syncing Live Context..."):
                common_news = fetch_news(f"{city_key} top headlines", 1)[0]
                health_news = fetch_news(f"{primary} healthcare trends India", 1)[0]
                live_weather = fetch_live_weather(city_key)

            # --- INTELLIGENCE CARD ---
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 2px solid #1e293b; padding: 25px; border-radius: 15px; color: #000; margin-bottom: 25px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h2 style="margin:0;">🕵️ {primary.upper()}</h2>
                            <span style="background: #e2e8f0; color: #334155; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: 600;">📖 {SEGMENT_DEFS.get(seg_key, 'General Healthcare Cohort')}</span>
                        </div>
                        <span style="background: #ef4444; color:#fff; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{live_weather} | {city_key.upper()}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                        <div style="background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd;"><b>📰 Common:</b> <a href="{common_news['link']}" target="_blank">{common_news['title']}</a></div>
                        <div style="background: #fff; padding: 12px; border-radius: 8px; border: 1px solid #ddd;"><b>🏥 Health:</b> <a href="{health_news['link']}" target="_blank">{health_news['title']}</a></div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align:center;">
                        <div style="background:#f1f5f9; padding:10px; border-radius:8px;">👵 Seniors: <b>{dna['seniors']}</b></div>
                        <div style="background:#f1f5f9; padding:10px; border-radius:8px;">🍼 Moms: <b>{dna['moms']}</b></div>
                        <div style="background:#f1f5f9; padding:10px; border-radius:8px;">👩 Female: <b>{dna['females']}</b></div>
                        <div style="background:#f1f5f9; padding:10px; border-radius:8px;">📱 Tech: <b>{dna['tech']}</b></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- 5. THE DYNAMIC CROSS-SELL MATRIX ---
            st.subheader("🛒 Professional Cross-Sell Matrix")
            
            # Category-Based Affinity Rules
            if "mom" in p_lower or "baby" in p_lower:
                base_data = [
                    ["Pampers New Baby Taped Diapers", "Apollo Life Wet Wipes", "High-frequency consumption match; 1:1 usage ratio."],
                    ["Nestle NAN PRO 2", "Morisons Feeding Bottle", "Health safety hook; formula users require sterile gear."],
                    ["Johnsons Baby Wash", "Sebamed Baby Lotion", "Complete post-bath skin moisture routine."],
                    ["Himalaya Diaper Rash Cream", "Mamaearth Baby Powder", "Dermatological protection bundle for infants."],
                    ["Silicone Teether", "Woodwards Gripe Water", "Teething leads to irritability; stomach relief is the logical next buy."]
                ]
            elif "cardio" in p_lower or "diab" in p_lower or "diag" in p_lower:
                base_data = [
                    ["OneTouch Select Plus", "OneTouch Lancets", "Essential consumables for every blood glucose check."],
                    ["Omron Blood Pressure", "Accu-Chek Active Kit", "Vital health kit; cardio patients monitor oxygen & pressure."],
                    ["GNC Fish Oil", "Apollo Life Multivitamin", "Cardiac nutritional foundation; heart-healthy lipid stack."],
                    ["Diabetic Socks", "Diabetic Foot Care Cream", "Neuropathy prevention; extremity care is high priority."],
                    ["Protinex Diabetes Care", "Sugar Free Gold", "Dietary conversion bundle for diabetic lifestyle."]
                ]
            elif "skin" in p_lower:
                base_data = [
                    ["Cetaphil Gentle Cleanser", "Cetaphil SPF 50", "Dermatologist routine; cleansers paired with photo-aging prevention."],
                    ["Minimalist Salicylic", "Plum Green Tea Toner", "Barrier repair necessity after chemical exfoliation."],
                    ["Garnier Micellar Water", "Bioderma Sensibio", "Premium upsell transition for urban cosmetic removal."],
                    ["Pears Body Wash", "Nivea Body Milk", "Bath utility and moisture-lock routine completion."],
                    ["Aloe Vera Gel", "Apollo Calamine", "Summer heatwave soothing bundle for irritated skin."]
                ]
            else: # Urban General
                base_data = [
                    ["ORS Orange", "Apollo Sunscreen SPF 50", "Heatwave defense; hydration paired with UV protection."],
                    ["Seven Seas Cod Liver", "Apollo Vitamin C Zinc", "Immunity anchor for office-going urbanites."],
                    ["Eno Lemon", "Probiotic Capsules", "Gut health restoration after seasonal acidity spikes."],
                    ["Dolo 650", "Vicks VapoRub", "Broad viral symptom kit; fever + respiratory relief."],
                    ["Indulekha Hair Oil", "Tresemme Keratin Shampoo", "Cosmetic treatment bundle for hard-water damage."]
                ]

            # APPLY SEGMENT OVERRIDES
            final_rows = []
            for row in base_data:
                purchase, push, reason = row
                if seg_key in ["churn", "winback"]:
                    reason = f"Reactivation Offer: {reason} + 25% Off Coupon."
                elif seg_key == "power":
                    reason = f"Loyalty Reward: {reason} | Auto-refill recommended."
                
                final_rows.append([apl_link(purchase, purchase), apl_link(push, push), reason])

            df_xsell = pd.DataFrame(final_rows, columns=["User Purchase", "Push (Linked)", "Reasoning"])
            st.markdown(df_xsell.to_html(escape=False, index=False), unsafe_allow_html=True)
            st.write("")

    # --- 6. AGGREGATED ROI FORECAST ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Aggregated Reach & ROI")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total", f"{int(stats['Total']):,}"); m2.metric("WA", f"{int(stats['WA']):,}"); m3.metric("Push", f"{int(stats['Push']):,}"); m4.metric("SMS", f"{int(stats['SMS']):,}"); m5.metric("Email", f"{int(stats.get('Email', 0)):,}")

    cv1, cv2, cv3 = st.columns(3)
    wa_rate = cv1.number_input("WA Cost", value=0.78)
    sms_rate = cv2.number_input("SMS Cost", value=0.13)
    email_rate = cv3.number_input("Email Cost", value=0.03)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conv Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("AOV (₹)", value=800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "0.0x"}

    st.table(pd.DataFrame([calc("Push", stats['Push'], 0.0), calc("WhatsApp", stats['WA'], wa_rate), calc("SMS", stats['SMS'], sms_rate), calc("Email", stats.get('Email', 0), email_rate)]))
