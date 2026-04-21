import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re

# --- 1. LIVE UTILITIES ---
@st.cache_data(ttl=300)
def fetch_news(query, count=1):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        items = root.findall('./channel/item')
        if items:
            return {"title": items[0].find('title').text.split(' - ')[0], "link": items[0].find('link').text}
    except: pass
    return {"title": "News Feed Offline", "link": "#"}

@st.cache_data(ttl=60)
def fetch_live_weather(city_key, fallback_string):
    """Fetches real-time weather using Open-Meteo API locked to IST."""
    coords = {
        "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
        "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639)
    }
    if city_key not in coords: return fallback_string
    lat, lon = coords[city_key]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FKolkata"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        if "current" in res:
            curr = res["current"]
            temp = curr.get("temperature_2m", "--")
            hum = curr.get("relative_humidity_2m", "--")
            code = curr.get("weather_code", 0)
            cond = "Clear" if code == 0 else "Rain" if code > 50 else "Cloudy"
            return f"{temp}°C | {cond} | Hum: {hum}%"
        return fallback_string
    except: return fallback_string

# --- 2. THE REAL-TIME AI GENERATOR (GEMINI 3 FLASH) ---
def generate_live_ai_xsell(sheet_context, target_name, segment_def, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Strategist for Apollo India. 
        City: {city}, Weather: {weather}, Context: {sheet_context}, Cohort: {target_name}, Def: {segment_def}. 
        Suggest 5 specific pharmacy cross-sell pairs based on these live conditions.
        Output ONLY a valid JSON array: [["Product A", "Product B", "Strategy Reason"]]
        """
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"AI Connection Error: {e}")
        return None

# --- 3. STATIC DEMOGRAPHICS & SEGMENTS ---
DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "31°C | Mist"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "36°C | Heat"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "31°C | Clear"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "35°C | Yellow Alert"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "30°C | Cloudy"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "30°C | Mist"}
}

SEGMENT_DEFS = {
    "ntu": "Non-Transacting Users (0 transactions in 60 days)",
    "churn": "Old users coming every 30 days and transacting",
    "winback": "Old NTU users coming back",
    "active": "Users with 1, 2, or 3 transactions only",
    "power": "Users hitting their 4th transaction",
    "enhancement": "High-volume users with many transactions"
}

def run_page():
    now = datetime.now()
    st.header("🛡️ Strategic Growth Predictor")
    st.caption(f"**Live Sync:** {now.strftime('%A, %d %B %Y | %I:%M %p')}")

    # --- DATA LOAD (SKU PARSER) ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    @st.cache_data
    def get_data():
        try:
            xl = pd.ExcelFile(EXCEL_URL, engine='openpyxl')
            rows = []
            for s in xl.sheet_names:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all')
                for i in range(0, len(df), 2):
                    if i + 1 >= len(df): break
                    r1, r2 = df.iloc[i], df.iloc[i+1]
                    if str(r1.iloc[0]).lower() in ['city', 'category', 'segment', 'nan']: continue
                    p_id, s_name = str(r1.iloc[0]).strip(), str(r2.iloc[0]).strip()
                    ui = f"{s_name} ({p_id})" if len(s_name) > 3 else p_id
                    ai = s_name if len(s_name) > 3 else p_id
                    rows.append({
                        'UI_Name': ui, 'AI_Name': ai, 'Sheet': s,
                        'Total': int(r1.iloc[1]), 'WA': int(r1.iloc[7]), 
                        'Push': int(r1.iloc[3]), 'SMS': int(r1.iloc[4]), 'Email': int(r1.iloc[5])
                    })
            return rows
        except: return []

    all_rows = get_data()

    # --- AGGRESSIVE BUCKETING (REGEX FIX FOR HYD/BLR) ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    city_pattern = re.compile(r'\b(hyd|blr|del|mum|chn|kol|ncr|hyderabad|bangalore|banglore|mumbai|delhi|chennai|kolkata)\b', re.I)

    for r in all_rows:
        n, s = r['UI_Name'].lower(), r['Sheet'].lower()
        if 'circle' in n or 'circle' in s: circle_rows.append(r)
        elif 'sku' in n or 'sku' in s: sku_rows.append(r)
        elif city_pattern.search(n) or 'city' in s: city_rows.append(r)
        elif 'daily' in s or 'portfolio' in s: daily_rows.append(r)
        else: focus_rows.append(r)

    # --- UI SELECTIONS ---
    st.markdown("### 📂 Selection Matrix")
    sel = []
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        p1 = st.multiselect("🏙️ City", [r['UI_Name'] for r in city_rows])
        for x in p1: sel.append(next(r for r in city_rows if r['UI_Name'] == x))
    with c2:
        p2 = st.multiselect("🎯 Focus Category", [r['UI_Name'] for r in focus_rows])
        for x in p2: sel.append(next(r for r in focus_rows if r['UI_Name'] == x))
    with c3:
        p3 = st.multiselect("💊 Daily Pharma", [r['UI_Name'] for r in daily_rows])
        for x in p3: sel.append(next(r for r in daily_rows if r['UI_Name'] == x))
    with c4:
        p4 = st.multiselect("⭐ Circle", [r['UI_Name'] for r in circle_rows])
        for x in p4: sel.append(next(r for r in circle_rows if r['UI_Name'] == x))
    with c5:
        p5 = st.multiselect("📦 SKU Based", [r['UI_Name'] for r in sku_rows])
        for x in p5: sel.append(next(r for r in sku_rows if r['UI_Name'] == x))

    if not sel:
        st.info("👋 Select cohorts above to activate live intelligence.")
        return

    # --- TABS & AI ENGINE ---
    st.divider()
    tabs = st.tabs([c['AI_Name'][:20] for c in sel])

    def apl_link(display_name):
        url = f"https://www.apollopharmacy.in/search-medicines/{display_name.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {display_name}</a>'

    for i, c in enumerate(sel):
        with tabs[i]:
            # Detect City Context
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in c['UI_Name'].lower() or k in c['UI_Name'].lower(): 
                    city_key = k
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])
            news = fetch_news(f"{city_key} health trends")
            seg_def = next((v for k,v in SEGMENT_DEFS.items() if k in c['UI_Name'].lower()), "General Patient")

            st.markdown(f"**Context:** {city_key.upper()} | {weather} | **Definition:** {seg_def}")
            st.info(f"📰 **News:** [{news['title']}]({news['link']})")
            
            # AI Auto-Generation
            api_key = st.secrets["GEMINI_API_KEY"]
            state_key = f"ai_{c['UI_Name']}"
            
            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("Generating Live Strategy..."):
                    st.session_state[state_key] = generate_live_ai_xsell(c['Sheet'], c['AI_Name'], seg_def, weather, city_key, api_key)

            if st.button("🔄 Refresh Strategy", key=f"btn_{i}"):
                st.session_state[state_key] = generate_live_ai_xsell(c['Sheet'], c['AI_Name'], seg_def, weather, city_key, api_key)

            if st.session_state[state_key]:
                formatted_rows = [[apl_link(r[0]), apl_link(r[1]), r[2]] for r in st.session_state[state_key]]
                df_out = pd.DataFrame(formatted_rows, columns=["Anchor", "Cross-Sell", "Reasoning"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- AGGREGATED ROI ---
    st.divider()
    t_base = sum(c['Total'] for c in sel)
    t_wa = sum(c['WA'] for c in sel)
    t_push = sum(c['Push'] for c in sel)
    t_sms = sum(c['SMS'] for c in sel)
    t_email = sum(c['Email'] for c in sel)

    st.subheader("🧬 Aggregated Reach & ROI Forecast")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{t_base:,}"); m2.metric("WhatsApp", f"{t_wa:,}"); m3.metric("Push", f"{t_push:,}"); m4.metric("SMS", f"{t_sms:,}"); m5.metric("Email", f"{t_email:,}")

    cv1, cv2, cv3 = st.columns(3)
    wa_r, sms_r, email_r = cv1.number_input("WA Cost", 0.78), cv2.number_input("SMS Cost", 0.13), cv3.number_input("Email Cost", 0.03)
    f1, f2 = st.columns(2)
    conv, aov = f1.slider("Conv Rate (%)", 0.1, 5.0, 1.0), f2.number_input("AOV (₹)", 800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "0.0x"}

    st.table(pd.DataFrame([calc("Push", t_push, 0.0), calc("WA", t_wa, wa_r), calc("SMS", t_sms, sms_r), calc("Email", t_email, email_r)]))

if __name__ == "__main__":
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    run_page()
