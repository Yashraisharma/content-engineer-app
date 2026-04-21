import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re
from io import BytesIO

# --- 1. CORE UTILITIES: WEATHER, NEWS, AI ---

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
    return {"title": "Intelligence Feed Offline", "link": "#"}

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
            return f"{c['temperature_2m']}°C | {cond} | Humidity: {c['relative_humidity_2m']}%"
    except: pass
    return fallback

def generate_live_ai_xsell(sheet, name, segment, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Role: Clinical Retail Strategist for Apollo Pharmacy India.
        Context: City: {city} | Weather: {weather} | Category: {sheet} | Target: {name} | Segment: {segment}.
        Task: Generate 5 specific pharmacy cross-sell pairs. 
        Format: Return ONLY a JSON array of arrays: [["Anchor", "Cross-Sell", "Strategic Reason"]]
        """
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except: return None

# --- 2. CONFIGURATION & DEMOGRAPHICS ---

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
    "churn": "Inactive users at risk of leaving",
    "winback": "Lapsed users being targeted for return",
    "active": "Standard active users (1-3 transactions)",
    "power": "Loyal users (4+ transactions)",
    "enhancement": "Premium high-volume users"
}

# --- 3. MAIN DASHBOARD ENGINE ---

def run_page():
    now = datetime.now()
    st.header("🛡️ Strategic Growth Predictor")
    st.caption(f"**Live Sync:** {now.strftime('%A, %d %B %Y | %I:%M %p')}")

    # --- DATA PARSER: Linked Row Logic ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    
    @st.cache_data
    def get_data():
        try:
            resp = requests.get(EXCEL_URL)
            xl = pd.ExcelFile(BytesIO(resp.content), engine='openpyxl')
            all_rows = []
            for s in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=s).dropna(how='all')
                # We iterate row by row to find ID -> Name pairs
                i = 0
                while i < len(df) - 1:
                    r1 = df.iloc[i]
                    r2 = df.iloc[i+1]
                    
                    id_val = str(r1.iloc[0]).strip()
                    name_val = str(r2.iloc[0]).strip()
                    
                    # Logic Check: If r1 is a header or empty, skip 1
                    if id_val.lower() in ['city', 'category', 'segment', 'nan', 'none']:
                        i += 1
                        continue
                    
                    # Logic Check: r1 is ID, r2 is Name. If r2 is nan, it's a single-row cohort
                    ui = f"{name_val} ({id_val})" if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    ai = name_val if len(name_val) > 2 and name_val.lower() != 'nan' else id_val
                    
                    try:
                        all_rows.append({
                            'UI_Name': ui, 'AI_Name': ai, 'Sheet': s,
                            'Total': int(float(r1.iloc[1])) if not pd.isna(r1.iloc[1]) else 0,
                            'WA': int(float(r1.iloc[7])) if not pd.isna(r1.iloc[7]) else 0,
                            'Push': int(float(r1.iloc[3])) if not pd.isna(r1.iloc[3]) else 0,
                            'SMS': int(float(r1.iloc[4])) if not pd.isna(r1.iloc[4]) else 0,
                            'Email': int(float(r1.iloc[5])) if not pd.isna(r1.iloc[5]) else 0
                        })
                        i += 2 # Move to next pair
                    except:
                        i += 1 # Something wrong with numbers, skip 1
            return all_rows
        except Exception as e:
            st.error(f"Excel Error: {e}")
            return []

    data = get_data()

    # --- BUCKETING LOGIC (CITY & CATEGORY SEGMENTATION) ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    
    # Comprehensive Regex for City Codes & Typos
    city_tag = re.compile(r'(?i)\b(hyd|hyderabad|hyderbad|hydewrabad|blr|bangalore|banglore|del|delhi|mum|mumbai|chn|chennai|kol|kolkata|ncr|bengaluru)\b')

    for r in data:
        n, s = r['UI_Name'].lower(), r['Sheet'].lower()
        
        # Priority 1: Circle
        if 'circle' in n or 'circle' in s:
            circle_rows.append(r)
        # Priority 2: City (Handles Hyd/Blr/Typos)
        elif city_tag.search(n) or 'city' in s:
            city_rows.append(r)
        # Priority 3: Daily Portfolio
        elif 'daily' in s or 'portfolio' in s:
            daily_rows.append(r)
        # Priority 4: SKU Specific
        elif 'sku' in n or 'sku' in s or r['AI_Name'] == r['UI_Name']:
            sku_rows.append(r)
        # Priority 5: Focus Category
        else:
            focus_rows.append(r)

    # --- UI: 5 CATEGORY SELECTION MATRIX ---
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

    if not sel:
        st.info("👋 Select cohorts above to activate live intelligence.")
        return

    # --- ENGINE TABS ---
    st.divider()
    tabs = st.tabs([c['AI_Name'][:22] for c in sel])

    def apl_link(val):
        url = f"https://www.apollopharmacy.in/search-medicines/{val.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {val}</a>'

    for i, cohort in enumerate(sel):
        with tabs[i]:
            # Context Detection (Fuzzy Hyderabad/Bangalore check)
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in cohort['UI_Name'].lower() or k in cohort['UI_Name'].lower(): city_key = k
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])
            news = fetch_news(f"{city_key} pharmacy healthcare trends")
            seg_def = next((v for k,v in SEGMENT_DEFS.items() if k in cohort['UI_Name'].lower()), "Active Customer")

            # UI Header Block
            st.markdown(f"### 🕵️ {cohort['UI_Name']}")
            st.markdown(f"**Location:** {city_key.upper()} | **Current Weather:** {weather}")
            st.info(f"📰 **Live Intelligence:** [{news['title']}]({news['link']})")

            # Demographic Metric Row
            st.write("#### 🧬 Cohort Demographics")
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("👵 Seniors", dna['seniors'])
            dc2.metric("🍼 Moms", dna['moms'])
            dc3.metric("👩 Female", dna['females'])
            dc4.metric("📱 Tech Savvy", dna['tech'])

            # AI Strategy Generation
            st.divider()
            st.subheader("🛒 Real-Time AI Strategy Matrix")
            state_key = f"ai_st_{cohort['UI_Name']}"
            api_key = st.secrets["GEMINI_API_KEY"]
            
            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("AI analyzing weather and segment for optimal cross-sell..."):
                    st.session_state[state_key] = generate_live_ai_xsell(cohort['Sheet'], cohort['AI_Name'], seg_def, weather, city_key, api_key)

            if st.button("🔄 Refresh Strategy", key=f"re_{i}"):
                st.session_state[state_key] = generate_live_ai_xsell(cohort['Sheet'], cohort['AI_Name'], seg_def, weather, city_key, api_key)

            if st.session_state[state_key]:
                formatted = [[apl_link(r[0]), apl_link(r[1]), r[2]] for r in st.session_state[state_key]]
                df_out = pd.DataFrame(formatted, columns=["Anchor Product", "Cross-Sell Product", "Strategic Reasoning"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- AGGREGATED ROI FORECAST ---
    st.divider()
    t_base, t_wa, t_push, t_sms, t_email = sum(c['Total'] for c in sel), sum(c['WA'] for c in sel), sum(c['Push'] for c in sel), sum(c['SMS'] for c in sel), sum(c['Email'] for c in sel)
    
    st.subheader("🧬 Aggregated ROI Forecast")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{t_base:,}")
    m2.metric("WhatsApp", f"{t_wa:,}")
    m3.metric("Mobile Push", f"{t_push:,}")
    m4.metric("SMS", f"{t_sms:,}")
    m5.metric("Email", f"{t_email:,}")

    cv1, cv2, cv3 = st.columns(3)
    wa_r, sms_r, email_r = cv1.number_input("WA Cost (₹)", value=0.78), cv2.number_input("SMS Cost (₹)", value=0.13), cv3.number_input("Email Cost (₹)", value=0.03)
    f1, f2 = st.columns(2)
    conv, aov = f1.slider("Conv Rate (%)", 0.1, 5.0, 1.0), f2.number_input("AOV (₹)", value=800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        roi = (rev/spend) if spend > 0 else 0
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{roi:.1f}x"}

    st.table(pd.DataFrame([
        calc("Push (Free)", t_push, 0.0), 
        calc("WhatsApp", t_wa, wa_r), 
        calc("SMS", t_sms, sms_r), 
        calc("Email", t_email, email_r)
    ]))

if __name__ == "__main__":
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    run_page()
