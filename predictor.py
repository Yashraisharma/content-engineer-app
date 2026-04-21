import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re
from io import BytesIO

# --- 1. CORE UTILITIES ---

@st.cache_data(ttl=300)
def fetch_news(query):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = root.findall('./channel/item')
        if items:
            return {"title": items[0].find('title').text.split(' - ')[0], "link": items[0].find('link').text}
    except: pass
    return {"title": "Live Intelligence Feed Offline", "link": "#"}

@st.cache_data(ttl=60)
def fetch_live_weather(city_key, fallback):
    coords = {
        "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
        "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639)
    }
    if city_key not in coords: return fallback
    lat, lon = coords[city_key]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FKolkata"
        res = requests.get(url, timeout=5).json()
        if "current" in res:
            c = res["current"]
            temp, hum, code = c.get("temperature_2m"), c.get("relative_humidity_2m"), c.get("weather_code", 0)
            cond = "☀️ Clear" if code == 0 else "🌧️ Rain" if code > 50 else "☁️ Cloudy"
            return f"{temp}°C | {cond} | Hum: {hum}%"
    except: pass
    return fallback

def generate_live_ai_xsell(sheet, name, segment, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Role: Senior Clinical Retail Strategist for Apollo Pharmacy India.
        Context: City: {city} | Weather: {weather} | Source Sheet: {sheet} | Target Item: {name} | Segment: {segment}.
        STRATEGIC CONSTRAINT:
        1. AVOID Rx/Hardcore medicines.
        2. PRIORITIZE High-Margin FMCG, Baby Care, Health Devices, and Nutrition.
        3. Logic: Suggest 5 pairs in JSON array: [["Anchor", "Cross-Sell", "Strategy Reason"]]
        """
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except: return None

# --- 2. CONFIG & STYLING ---

DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "31°C | Mist"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "36°C | Heat Alert"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "31°C | Clear"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "35°C | Yellow Alert"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "30°C | Cloudy"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "30°C | Mist"}
}

SEGMENT_DEFS = {
    "ntu": "Non-Transacting Users (0 transactions in 60 days)",
    "churn": "Inactive users at risk of leaving",
    "winback": "Lapsed users being targeted for return",
    "active": "Standard active users (1-3 transactions)",
    "power": "Loyal users (4+ transactions)",
    "enhancement": "Premium high-volume users"
}

CSS = """
<style>
    .main { background-color: #f8fafc; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .news-card { padding: 20px; border-radius: 12px; margin-bottom: 10px; border-left: 5px solid; }
    .common-news { background-color: #eff6ff; border-left-color: #3b82f6; }
    .health-news { background-color: #f0fdf4; border-left-color: #22c55e; }
    .strategy-table { border-collapse: collapse; margin: 25px 0; font-size: 0.9em; min-width: 400px; border-radius: 5px 5px 0 0; overflow: hidden; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }
    .roi-section { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #e2e8f0; margin-top: 30px; }
    h3 { color: #1e293b; font-weight: 700; }
</style>
"""

# --- 3. DASHBOARD ENGINE ---

def run_page():
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    
    now = datetime.now()
    t1, t2 = st.columns([3, 1])
    with t1:
        st.title("🛡️ Strategic Growth Predictor")
        st.markdown(f"**Live Dashboard Status:** Active Sync @ {now.strftime('%I:%M %p')}")
    with t2:
        st.image("https://www.apollopharmacy.in/static/images/logo.svg", width=180)

    # --- DATA PARSER ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    
    @st.cache_data
    def get_data():
        try:
            resp = requests.get(EXCEL_URL)
            xl = pd.ExcelFile(BytesIO(resp.content), engine='openpyxl')
            all_rows = []
            for s in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=s).dropna(how='all')
                for i in range(0, len(df) - 1, 2):
                    r1, r2 = df.iloc[i], df.iloc[i+1]
                    id_val, name_val = str(r1.iloc[0]).strip(), str(r2.iloc[0]).strip()
                    if id_val.lower() in ['city', 'category', 'segment', 'nan']: continue
                    ui = f"{name_val} ({id_val})" if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    ai = name_val if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    try:
                        all_rows.append({
                            'UI_Name': ui, 'AI_Name': ai, 'Sheet_Key': s,
                            'Total': int(float(r1.iloc[1])) if not pd.isna(r1.iloc[1]) else 0,
                            'WA': int(float(r1.iloc[7])) if not pd.isna(r1.iloc[7]) else 0,
                            'Push': int(float(r1.iloc[3])) if not pd.isna(r1.iloc[3]) else 0,
                            'SMS': int(float(r1.iloc[4])) if not pd.isna(r1.iloc[4]) else 0,
                            'Email': int(float(r1.iloc[5])) if not pd.isna(r1.iloc[5]) else 0
                        })
                        i += 2 
                    except: i += 1
            return all_rows
        except: return []

    all_data = get_data()

    # --- BUCKETING ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    city_regex = re.compile(r'(^|[\s\-_])(hyd|hyderabad|blr|bangalore|banglore|del|delhi|mum|mumbai|chn|chennai|kol|kolkata|ncr|bengaluru)([\s\-_]|$)', re.I)

    for r in all_data:
        n, s = r['UI_Name'].lower(), r['Sheet_Key'].lower()
        if 'circle' in n: circle_rows.append(r)
        elif s == 'top 6 cities' or city_regex.search(n): city_rows.append(r)
        elif 'pharma_focus' in s: focus_rows.append(r)
        elif 'daily_pharma' in s: daily_rows.append(r)
        else: sku_rows.append(r)

    # --- SELECTION MATRIX ---
    with st.container():
        st.markdown("### 📂 Segment Selector")
        c1, c2, c3, c4, c5 = st.columns(5)
        sel = []
        with c1:
            p1 = st.multiselect("🏙️ City", options=[r['UI_Name'] for r in city_rows])
            for x in p1: sel.append(next(r for r in city_rows if r['UI_Name'] == x))
        with c2:
            p2 = st.multiselect("🎯 Focus", options=[r['UI_Name'] for r in focus_rows])
            for x in p2: sel.append(next(r for r in focus_rows if r['UI_Name'] == x))
        with c3:
            p3 = st.multiselect("💊 Portfolio", options=[r['UI_Name'] for r in daily_rows])
            for x in p3: sel.append(next(r for r in daily_rows if r['UI_Name'] == x))
        with c4:
            p4 = st.multiselect("⭐ Circle", options=[r['UI_Name'] for r in circle_rows])
            for x in p4: sel.append(next(r for r in circle_rows if r['UI_Name'] == x))
        with c5:
            p5 = st.multiselect("📦 SKU", options=[r['UI_Name'] for r in sku_rows])
            for x in p5: sel.append(next(r for r in sku_rows if r['UI_Name'] == x))

    if not sel:
        st.info("👋 Select cohorts above to launch Growth Analysis.")
        return

    # --- TABS ---
    st.divider()
    tabs = st.tabs([f"📊 {c['AI_Name'][:18]}" for c in sel])

    for i, cohort in enumerate(sel):
        with tabs[i]:
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in cohort['UI_Name'].lower() or k in cohort['UI_Name'].lower(): 
                    city_key = k; break
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])
            seg_def = next((v for k,v in SEGMENT_DEFS.items() if k in cohort['UI_Name'].lower()), "General Pharma Cohort")

            # News Cards
            n1, n2 = st.columns(2)
            common = fetch_news(f"{city_key} health trends")
            health = fetch_news(f"{cohort['AI_Name']} pharmacy")
            n1.markdown(f'<div class="news-card common-news"><b>🌐 Region Trends:</b><br><a href="{common["link"]}" target="_blank">{common["title"]}</a></div>', unsafe_allow_html=True)
            n2.markdown(f'<div class="news-card health-news"><b>🏥 Health Pulse:</b><br><a href="{health["link"]}" target="_blank">{health["title"]}</a></div>', unsafe_allow_html=True)

            # Metrics
            st.markdown(f"#### 🧬 {city_key.upper()} Context | {weather}")
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("👵 Seniors", dna['seniors'])
            dc2.metric("🍼 Moms", dna['moms'])
            dc3.metric("👩 Female", dna['females'])
            dc4.metric("📱 Tech Savvy", dna['tech'])

            # AI Table
            st.divider()
            state_key = f"ai_st_{cohort['UI_Name']}"
            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("Analyzing high-margin FMCG pairs..."):
                    st.session_state[state_key] = generate_live_ai_xsell(cohort['Sheet_Key'], cohort['AI_Name'], seg_def, weather, city_key, st.secrets["GEMINI_API_KEY"])

            st.subheader("🛒 Real-Time AI Strategy Matrix (FMCG Focus)")
            if st.session_state[state_key]:
                def apl(v): return f'<a href="https://www.apollopharmacy.in/search-medicines/{v.replace(" ","%20")}" target="_blank" style="color:#2563eb;text-decoration:none;font-weight:600;">🛒 {v}</a>'
                formatted = [[apl(r[0]), apl(r[1]), r[2]] for r in st.session_state[state_key]]
                df_out = pd.DataFrame(formatted, columns=["Anchor Product", "Cross-Sell (FMCG/Wellness)", "Strategic Reasoning"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.button("🔄 Regenerate Strategy", key=f"re_{i}")

    # --- ROI SECTION ---
    st.markdown('<div class="roi-section">', unsafe_allow_html=True)
    st.subheader("🧬 Aggregated Reach & ROI Forecast")
    t_base, t_wa, t_push, t_sms, t_email = sum(c['Total'] for c in sel), sum(c['WA'] for c in sel), sum(c['Push'] for c in sel), sum(c['SMS'] for c in sel), sum(c['Email'] for c in sel)
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{t_base:,}")
    m2.metric("WhatsApp", f"{t_wa:,}")
    m3.metric("Mobile Push", f"{t_push:,}")
    m4.metric("SMS", f"{t_sms:,}")
    m5.metric("Email", f"{t_email:,}")

    st.markdown("#### ⚙️ Campaign Parameters")
    cv1, cv2, cv3, cv4, cv5 = st.columns(5)
    wa_r = cv1.number_input("WA Cost (₹)", 0.78)
    sms_r = cv2.number_input("SMS Cost (₹)", 0.13)
    em_r = cv3.number_input("Email Cost (₹)", 0.03)
    conv = cv4.slider("Conv Rate (%)", 0.1, 5.0, 1.0)
    aov = cv5.number_input("AOV (₹)", 800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        roi = (rev/spend) if spend > 0 else 0
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{roi:.1f}x"}

    st.table(pd.DataFrame([calc("Push (Free)", t_push, 0.0), calc("WhatsApp", t_wa, wa_r), calc("SMS", t_sms, sms_r), calc("Email", t_email, em_r)]))
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run_page()
