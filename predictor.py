import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re
import numpy as np

# --- 1. CORE UTILITIES: WEATHER, NEWS, AI ---

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
    coords = {
        "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
        "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639)
    }
    if city_key not in coords: return fallback_string
    lat, lon = coords[city_key]
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FKolkata"
        res = requests.get(url, timeout=5).json()
        if "current" in res:
            curr = res["current"]
            temp, hum, code = curr.get("temperature_2m"), curr.get("relative_humidity_2m"), curr.get("weather_code", 0)
            cond = "Clear" if code == 0 else "Partly Cloudy" if code < 45 else "Rain" if code > 50 else "Mist"
            return f"{temp}°C | {cond} | Humidity: {hum}%"
        return fallback_string
    except: return fallback_string

def generate_live_ai_xsell(sheet, name, segment, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Role: Clinical Retail Strategist for Apollo Pharmacy India.
        Context: City: {city} | Weather: {weather} | Category: {sheet} | Target: {name} | Segment: {segment}.
        Task: Suggest 5 highly logical pharmacy cross-sell pairs. 
        Note: Focus on current weather and segment definition. 
        Format: Respond ONLY with a JSON array of arrays: [["Anchor", "Cross-Sell", "Reasoning"]]
        """
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

# --- 2. CONFIGURATION DATA ---

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
    "churn": "Old users coming every 30 days and transacting",
    "winback": "Old NTU users coming back",
    "active": "Users with 1, 2, or 3 transactions only",
    "power": "Users hitting their 4th transaction",
    "enhancement": "High-volume users with many transactions"
}

# --- 3. THE DASHBOARD ENGINE ---

def run_page():
    now = datetime.now()
    st.header("🛡️ Strategic Growth Predictor")
    st.caption(f"**Live Sync:** {now.strftime('%A, %d %B %Y | %I:%M %p')}")

    # --- DATA PARSER: SKU + MULTI-SHEET ---
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
                    
                    id_val, name_val = str(r1.iloc[0]).strip(), str(r2.iloc[0]).strip()
                    ui = f"{name_val} ({id_val})" if len(name_val) > 2 else id_val
                    ai = name_val if len(name_val) > 2 else id_val
                    
                    rows.append({
                        'UI_Name': ui, 'AI_Name': ai, 'Sheet': s,
                        'Total': int(r1.iloc[1]), 'WA': int(r1.iloc[7]), 
                        'Push': int(r1.iloc[3]), 'SMS': int(r1.iloc[4]), 'Email': int(r1.iloc[5])
                    })
            return rows
        except: return []

    all_data = get_data()

    # --- ADVANCED BUCKETING (HYD/BLR REGEX FIX) ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    
    # This Regex captures abbreviations, full names, and common typos like 'hydewrabad' or 'banglore'
    city_pattern = re.compile(r'\b(hyd|blr|del|mum|chn|kol|ncr|hyderabad|bangalore|banglore|hydewrabad|hyderbad|mumbai|delhi|chennai|kolkata|bengaluru)\b', re.I)

    for r in all_data:
        n, s = r['UI_Name'].lower(), r['Sheet'].lower()
        if 'circle' in n or 'circle' in s: circle_rows.append(r)
        elif 'sku' in n or 'sku' in s: sku_rows.append(r)
        elif city_pattern.search(n) or 'city' in s: city_rows.append(r)
        elif 'daily' in s or 'portfolio' in s: daily_rows.append(r)
        else: focus_rows.append(r)

    # --- UI SELECTION MATRIX ---
    st.markdown("### 📂 Target Selection Matrix")
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

    # --- TABS & CONTENT GENERATION ---
    st.divider()
    tabs = st.tabs([c['AI_Name'][:22] for c in sel])

    def apl_link(val):
        url = f"https://www.apollopharmacy.in/search-medicines/{val.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {val}</a>'

    for i, cohort in enumerate(sel):
        with tabs[i]:
            # Context Detection
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in cohort['UI_Name'].lower() or k in cohort['UI_Name'].lower(): city_key = k
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])
            news = fetch_news(f"{city_key} health trends")
            seg_def = next((v for k,v in SEGMENT_DEFS.items() if k in cohort['UI_Name'].lower()), "General Patient")

            # UI Header Block
            st.markdown(f"### 🕵️ {cohort['UI_Name']}")
            st.markdown(f"**Origin Category:** {cohort['Sheet']} | **Definition:** {seg_def}")
            st.markdown(f"**Location:** {city_key.upper()} ({weather}) [🔗 AccuWeather](https://www.accuweather.com/en/search-locations?query={city_key})")
            
            nc1, nc2 = st.columns(2)
            nc1.info(f"**📰 Common News:** [{news['title']}]({news['link']})")
            nc2.success(f"**🏥 Health News:** [Active Trend Discovery]({news['link']})")

            # Demographics Block
            st.write("#### 🧬 Cohort Demographics")
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("👵 Seniors", dna['seniors'])
            dc2.metric("🍼 Moms", dna['moms'])
            dc3.metric("👩 Female", dna['females'])
            dc4.metric("📱 Tech Savvy", dna['tech'])

            # AI Strategy Matrix
            st.divider()
            st.subheader("🛒 Real-Time AI Strategy Matrix")
            state_key = f"ai_st_{cohort['UI_Name']}"
            api_key = st.secrets["GEMINI_API_KEY"]

            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("AI Brainstorming unique combinations..."):
                    st.session_state[state_key] = generate_live_ai_xsell(cohort['Sheet'], cohort['AI_Name'], seg_def, weather, city_key, api_key)

            if st.button("🔄 Refresh Strategy", key=f"btn_re_{i}"):
                st.session_state[state_key] = generate_live_ai_xsell(cohort['Sheet'], cohort['AI_Name'], seg_def, weather, city_key, api_key)

            if st.session_state[state_key]:
                formatted = [[apl_link(r[0]), apl_link(r[1]), r[2]] for r in st.session_state[state_key]]
                df_out = pd.DataFrame(formatted, columns=["Anchor Product", "Cross-Sell Product", "Strategic Reasoning"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- AGGREGATED ROI FORECAST ---
    st.divider()
    t_base = sum(c['Total'] for c in sel)
    t_wa = sum(c['WA'] for c in sel)
    t_push = sum(c['Push'] for c in sel)
    t_sms = sum(c['SMS'] for c in sel)
    t_email = sum(c['Email'] for c in sel)

    st.subheader("🧬 Aggregated Reach & ROI Forecast")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{t_base:,}")
    m2.metric("WhatsApp", f"{t_wa:,}")
    m3.metric("Mobile Push", f"{t_push:,}")
    m4.metric("SMS", f"{t_sms:,}")
    m5.metric("Email", f"{t_email:,}")

    cv1, cv2, cv3 = st.columns(3)
    wa_rate = cv1.number_input("WA Cost (₹)", value=0.78)
    sms_rate = cv2.number_input("SMS Cost (₹)", value=0.13)
    email_rate = cv3.number_input("Email Cost (₹)", value=0.03)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Avg Order Value (₹)", value=800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        roi = (rev/spend) if spend > 0 else 0
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{roi:.1f}x"}

    st.table(pd.DataFrame([
        calc("Push (Free)", t_push, 0.0), 
        calc("WhatsApp", t_wa, wa_rate), 
        calc("SMS", t_sms, sms_rate), 
        calc("Email", t_email, email_rate)
    ]))

if __name__ == "__main__":
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    run_page()
