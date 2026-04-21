import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re
from io import BytesIO

# --- 1. LIVE UTILITIES ---

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
    return {"title": "News Feed Offline", "link": "#"}

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
            cond = "Clear" if c['weather_code'] == 0 else "Rain" if c['weather_code'] > 50 else "Cloudy"
            return f"{c['temperature_2m']}°C | {cond} | Hum: {c['relative_humidity_2m']}%"
    except: pass
    return fallback

def generate_live_ai_xsell(sheet, name, segment, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Role: Clinical Retail Strategist for Apollo Pharmacy India.
        Context: City: {city} | Weather: {weather} | Category: {sheet} | Target: {name} | Segment: {segment}.
        Generate 5 logical pharmacy cross-sell product pairs. Focus on current conditions.
        Output ONLY a JSON array of arrays: [["Anchor", "Cross-Sell", "Reasoning"]]
        """
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except: return None

# --- 2. CONFIG & ASSETS ---

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

# --- 3. MAIN APP ---

def run_page():
    st.header("🛡️ Strategic Growth Predictor")
    
    # --- SMART DATA PARSER (SKU Logic) ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    
    @st.cache_data
    def get_data():
        try:
            resp = requests.get(EXCEL_URL)
            xl = pd.ExcelFile(BytesIO(resp.content), engine='openpyxl')
            rows = []
            for s in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=s).dropna(how='all')
                for i in range(0, len(df) - 1, 2):
                    r1, r2 = df.iloc[i], df.iloc[i+1]
                    if str(r1.iloc[0]).lower() in ['city', 'category', 'segment', 'nan']: continue
                    
                    id_val, name_val = str(r1.iloc[0]).strip(), str(r2.iloc[0]).strip()
                    ui = f"{name_val} ({id_val})" if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    ai = name_val if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    
                    try:
                        rows.append({
                            'UI_Name': ui, 'AI_Name': ai, 'Sheet': s,
                            'Total': int(float(r1.iloc[1])) if not pd.isna(r1.iloc[1]) else 0,
                            'WA': int(float(r1.iloc[7])) if not pd.isna(r1.iloc[7]) else 0,
                            'Push': int(float(r1.iloc[3])) if not pd.isna(r1.iloc[3]) else 0,
                            'SMS': int(float(r1.iloc[4])) if not pd.isna(r1.iloc[4]) else 0,
                            'Email': int(float(r1.iloc[5])) if not pd.isna(r1.iloc[5]) else 0
                        })
                    except: continue
            return rows
        except Exception as e:
            st.error(f"Data Load Error: {e}")
            return []

    data = get_data()
    if not data:
        st.warning("⚠️ Data pool is empty. Please check the Excel file.")
        return

    # --- ENHANCED BUCKETING ENGINE (HYDERABAD FIX) ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    
    # Matches codes even if they are part of a string like 'Hyderabad_Segment' or 'Hyd-Active'
    # Prevents matching 'rehydration' by checking for separators or start/end
    city_regex = re.compile(r'(^|[\s\-_])(hyd|hyderabad|blr|bangalore|banglore|del|delhi|mum|mumbai|chn|chennai|kol|kolkata|ncr|bengaluru)([\s\-_]|$)', re.I)

    for r in data:
        n, s = r['UI_Name'].lower(), r['Sheet'].lower()
        
        if 'circle' in n or 'circle' in s:
            circle_rows.append(r)
        elif 'sku' in n or 'sku' in s:
            sku_rows.append(r)
        elif 'city' in s or city_regex.search(n):
            city_rows.append(r)
        elif 'daily' in s or 'portfolio' in s:
            daily_rows.append(r)
        else:
            focus_rows.append(r)

    # --- UI SELECTION ---
    st.markdown("### 📂 Target Selection Matrix")
    sel = []
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        opts = {r['UI_Name']: r for r in city_rows}
        p1 = st.multiselect("🏙️ City", options=list(opts.keys()))
        for x in p1: sel.append(opts[x])
    with c2:
        opts = {r['UI_Name']: r for r in focus_rows}
        p2 = st.multiselect("🎯 Focus Category", options=list(opts.keys()))
        for x in p2: sel.append(opts[x])
    with c3:
        opts = {r['UI_Name']: r for r in daily_rows}
        p3 = st.multiselect("💊 Daily Pharma", options=list(opts.keys()))
        for x in p3: sel.append(opts[x])
    with c4:
        opts = {r['UI_Name']: r for r in circle_rows}
        p4 = st.multiselect("⭐ Circle", options=list(opts.keys()))
        for x in p4: sel.append(opts[x])
    with c5:
        opts = {r['UI_Name']: r for r in sku_rows}
        p5 = st.multiselect("📦 SKU Based", options=list(opts.keys()))
        for x in p5: sel.append(opts[x])

    if not sel: return

    # --- GENERATION ---
    st.divider()
    tabs = st.tabs([c['AI_Name'][:22] for c in sel])
    for i, c in enumerate(sel):
        with tabs[i]:
            # City Context Detection
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in c['UI_Name'].lower() or k in c['UI_Name'].lower(): city_key = k
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])
            news = fetch_news(f"{city_key} pharmacy health")
            seg_def = next((v for k,v in SEGMENT_DEFS.items() if k in c['UI_Name'].lower()), "General Cohort")

            st.markdown(f"### 🕵️ {c['UI_Name']}")
            st.markdown(f"**Origin:** {c['Sheet']} | **Location:** {city_key.upper()} ({weather})")
            st.info(f"📰 **Latest News:** [{news['title']}]({news['link']})")

            # AI Logic
            state_key = f"ai_st_{c['UI_Name']}"
            api_key = st.secrets["GEMINI_API_KEY"]
            
            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("AI analyzing cohort and conditions..."):
                    st.session_state[state_key] = generate_live_ai_xsell(c['Sheet'], c['AI_Name'], seg_def, weather, city_key, api_key)

            if st.button("🔄 Refresh Strategy", key=f"re_{i}"):
                st.session_state[state_key] = generate_live_ai_xsell(c['Sheet'], c['AI_Name'], seg_def, weather, city_key, api_key)

            if st.session_state[state_key]:
                def apl(v): return f'<a href="https://www.apollopharmacy.in/search-medicines/{v.replace(" ","%20")}" target="_blank">🛒 {v}</a>'
                df_out = pd.DataFrame([[apl(r[0]), apl(r[1]), r[2]] for r in st.session_state[state_key]], columns=["Anchor", "Cross-Sell", "Strategy"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- AGGREGATED ROI ---
    st.divider()
    t_base, t_wa, t_push, t_sms, t_email = sum(c['Total'] for c in sel), sum(c['WA'] for c in sel), sum(c['Push'] for c in sel), sum(c['SMS'] for c in sel), sum(c['Email'] for c in sel)
    st.subheader("🧬 Aggregated Reach")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Base", f"{t_base:,}"); m2.metric("WA", f"{t_wa:,}"); m3.metric("Push", f"{t_push:,}"); m4.metric("SMS", f"{t_sms:,}"); m5.metric("Email", f"{t_email:,}")

if __name__ == "__main__":
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    run_page()
