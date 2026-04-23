import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai
import re
from io import BytesIO

# --- 1. CORE INTELLIGENCE: WEATHER, NEWS, AI ---

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
    return {"title": "Market intelligence feed currently offline", "link": "#"}

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

def generate_growth_strategy(sheet, name, segment, weather, city, api_key):
    """
    STRICT 2026 FLASH 3 ENGINE.
    No legacy models. 
    """
    if not api_key: 
        return [["Config Error", "API Key Missing", "Check Secrets for GEMINI_API_KEY"]]
    
    genai.configure(api_key=api_key)
    
    # 2026 Production & Preview strings for Gemini 3 series ONLY
    model_names = ['gemini-3-flash-preview', 'gemini-3-flash', 'gemini-3.1-flash-lite-preview']
    last_error = ""

    prompt = f"""
    Role: Senior Clinical Retail Strategist for Apollo Pharmacy India.
    Anchor: {name} ({sheet}) | Segment: {segment} | Env: {city}, {weather}

    STRATEGIC TASK: Suggest 5 hyper-logical growth pairings.
    BUSINESS RULES:
    1. CORE FOCUS: FMCG, Baby Care (Diapers/Nutrition), Hygiene (Microbiome repair/Antifungal textile), Devices (BP/Pulse), and Clinical Nutrition.
    2. EXCLUSION: NEVER suggest Rx (Prescription) or hardcore medicines.
    3. LOGIC: Complement medical treatment with high-margin wellness drivers (e.g., Moisture management for fungus, barrier repair for dermatologicals).
    
    Format: Respond ONLY with a valid JSON array: [["Anchor", "Cross-Sell", "Growth Logic"]]
    """
    
    for m_name in model_names:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            match = re.search(r'\[\s*\[.*\]\s*\]', response.text, re.DOTALL)
            if match: return json.loads(match.group())
        except Exception as e:
            last_error = str(e)
            continue
            
    return [["System Error", "Flash 3 Unreachable", f"Reason: {last_error}"]]

# --- 2. THEMES & CUSTOM STYLING (Blue/Black/White High Contrast) ---

def apply_custom_theme(mode):
    # Professional Blue/Black/White Palette
    if mode == "Midnight (Dark)":
        bg, text, card, accent, link = "#000000", "#f8fafc", "#111827", "#3b82f6", "#60a5fa"
    else:
        bg, text, card, accent, link = "#ffffff", "#000000", "#f9fafb", "#2563eb", "#1d4ed8"
    
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; }}
        
        /* Force Text Colors for Readability */
        h1, h2, h3, h4, p, span, label, div, .stTabs [data-baseweb="tab"] {{ 
            color: {text} !important; 
        }}
        
        /* High Contrast Metrics */
        [data-testid="stMetricLabel"] {{ color: {text} !important; font-weight: 700 !important; opacity: 0.9; font-size: 1.05rem; }}
        [data-testid="stMetricValue"] {{ color: {text} !important; font-weight: 800 !important; }}
        
        div[data-testid="stMetric"] {{ 
            background-color: {card}; 
            border: 1.5px solid {accent}33; 
            padding: 20px; 
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        .news-box {{ 
            background-color: {card}; 
            border-left: 5px solid {accent}; 
            padding: 15px; 
            border-radius: 8px; 
            margin-bottom: 12px; 
        }}

        a {{ color: {link} !important; font-weight: 700; text-decoration: underline; }}
        
        .roi-panel {{ 
            background-color: {card}; 
            padding: 30px; 
            border-radius: 15px; 
            border: 2px solid {accent}; 
            margin-top: 25px; 
        }}
        
        /* Table Styling */
        table {{ color: {text} !important; background-color: {card}; width: 100%; border-radius: 8px; overflow: hidden; }}
        th {{ background-color: {accent}22; padding: 12px; border-bottom: 2px solid {accent}44; text-align: left; font-weight: 800; }}
        td {{ padding: 12px; border-bottom: 1px solid {accent}11; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. MAIN DASHBOARD ENGINE ---

def run_page():
    st.set_page_config(page_title="Apollo Growth Strategist", layout="wide")
    
    with st.sidebar:
        st.image("https://www.apollopharmacy.in/static/images/logo.svg", width=150)
        theme_choice = st.radio("UI Theme:", ["Professional (Light)", "Midnight (Dark)"])
        st.divider()
        st.caption("Strategic Growth Predictor | Gemini 3 Flash v4.4")

    apply_custom_theme(theme_choice)
    now = datetime.now()
    st.title("🛡️ Strategic Growth Predictor")
    st.markdown(f"**Enterprise Pulse:** {now.strftime('%A, %d %B | %I:%M %p')}")

    # --- DATA PARSER ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    
    @st.cache_data
    def load_data():
        try:
            resp = requests.get(EXCEL_URL)
            xl = pd.ExcelFile(BytesIO(resp.content), engine='openpyxl')
            combined = []
            for sheet in xl.sheet_names:
                df = pd.read_excel(xl, sheet_name=sheet).dropna(how='all')
                i = 0
                while i < len(df) - 1:
                    r1, r2 = df.iloc[i], df.iloc[i+1]
                    id_raw, name_raw = str(r1.iloc[0]).strip(), str(r2.iloc[0]).strip()
                    if id_raw.lower() in ['city', 'category', 'segment', 'nan']:
                        i += 1; continue
                    ui_label = f"{name_raw} ({id_raw})" if len(name_raw) > 2 and name_raw.lower() != 'nan' else id_raw
                    ai_query = name_raw if len(name_raw) > 2 and name_raw.lower() != 'nan' else id_raw
                    try:
                        combined.append({
                            'UI_Name': ui_label, 'AI_Name': ai_query, 'Source_Sheet': sheet,
                            'Total': int(float(r1.iloc[1])), 'WA': int(float(r1.iloc[7])), 
                            'Push': int(float(r1.iloc[3])), 'SMS': int(float(r1.iloc[4])), 'Email': int(float(r1.iloc[5]))
                        })
                        i += 2 
                    except: i += 1
            return combined
        except: return []

    data = load_data()

    # --- SEGMENTATION (AGGRESSIVE CITY DETECTION) ---
    city_r, focus_r, port_r, circ_r, sku_r = [], [], [], [], []
    city_regex = re.compile(r'\b(hyd|hyderabad|hyderbad|hydewrabad|blr|bangalore|banglore|del|delhi|mum|mumbai|chn|chennai|kol|kolkata|ncr|bengaluru)\b', re.I)

    for c in data:
        n, s = c['UI_Name'].lower(), c['Source_Sheet'].lower()
        if 'circle' in n: circ_r.append(c)
        elif 'top 6 cities' in s or city_regex.search(n): city_r.append(c)
        elif 'pharma_focus' in s: focus_r.append(c)
        elif 'daily_pharma' in s: port_r.append(c)
        else: sku_r.append(c)

    # --- SELECTION HUB ---
    st.markdown("### 📂 Selection Matrix")
    c1, c2, c3, c4, c5 = st.columns(5)
    sel = []
    with c1:
        p = st.multiselect("🏙️ Cities", options=[x['UI_Name'] for x in city_r])
        for v in p: sel.append(next(x for x in city_r if x['UI_Name'] == v))
    with c2:
        p = st.multiselect("🎯 Focus", options=[x['UI_Name'] for x in focus_r])
        for v in p: sel.append(next(x for x in focus_r if x['UI_Name'] == v))
    with c3:
        p = st.multiselect("💊 Portfolio", options=[x['UI_Name'] for x in port_r])
        for v in p: sel.append(next(x for x in port_r if x['UI_Name'] == v))
    with c4:
        p = st.multiselect("⭐ Circle", options=[x['UI_Name'] for x in circ_r])
        for v in p: sel.append(next(x for x in circ_r if x['UI_Name'] == v))
    with c5:
        p = st.multiselect("📦 SKU", options=[x['UI_Name'] for x in sku_r])
        for v in p: sel.append(next(x for x in sku_r if x['UI_Name'] == v))

    if not sel:
        st.info("👋 Select cohorts above to launch analysis."); return

    # --- GROWTH TABS ---
    st.divider()
    tabs = st.tabs([f"📈 {c['AI_Name'][:18]}" for c in sel])

    DEMOGRAPHICS = {
        "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "31°C | Mist"},
        "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "36°C | Heat Alert"},
        "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "31°C | Clear"},
        "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "35°C | Yellow Alert"},
        "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "30°C | Cloudy"},
        "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "30°C | Mist"}
    }

    for i, cohort in enumerate(sel):
        with tabs[i]:
            city_key = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in cohort['UI_Name'].lower() or k in cohort['UI_Name'].lower(): city_key = k; break
            
            dna = DEMOGRAPHICS[city_key]
            weather = fetch_live_weather(city_key, dna['fallback'])

            # Live Context
            n1, n2 = st.columns(2)
            n1.markdown(f'<div class="news-box"><b>🌐 Local News:</b> {fetch_news(city_key)["title"]}</div>', unsafe_allow_html=True)
            n2.markdown(f'<div class="news-box"><b>🏥 Health Pulse:</b> {fetch_news(cohort["AI_Name"])["title"]}</div>', unsafe_allow_html=True)

            # Metrics (Black Text forced in CSS)
            st.markdown(f"#### 🧬 {city_key.upper()} Snapshot | {weather}")
            dc1, dc2, dc3, dc4 = st.columns(4)
            dc1.metric("👵 Seniors", dna['seniors'])
            dc2.metric("🍼 Moms", dna['moms'])
            dc3.metric("👩 Female", dna['females'])
            dc4.metric("📱 Tech Savvy", dna['tech'])

            # AI Cross-Sell Matrix
            st.divider()
            st.subheader("🛒 Strategic Cross-Sell Synthesis (FMCG Focus)")
            state_key = f"strat_flash3_{cohort['UI_Name']}"
            if state_key not in st.session_state:
                with st.spinner("Flash 3 generating wellness strategy..."):
                    st.session_state[state_key] = generate_growth_strategy(cohort['Source_Sheet'], cohort['AI_Name'], "Active", weather, city_key, st.secrets.get("GEMINI_API_KEY"))

            if st.session_state[state_key]:
                def link(v): return f'<a href="https://www.apollopharmacy.in/search-medicines/{v.replace(" ","%20")}" target="_blank">🛒 {v}</a>'
                df_out = pd.DataFrame([[link(r[0]), link(r[1]), r[2]] for r in st.session_state[state_key]], columns=["Anchor", "Pairing", "Growth Logic"])
                st.markdown(df_out.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.button("🔄 Refresh Data", key=f"re_btn_{i}")

    # --- ROI PANEL ---
    st.markdown('<div class="roi-panel">', unsafe_allow_html=True)
    st.subheader("🧬 Aggregated Campaign Impact")
    sums = {"Base": sum(c['Total'] for c in sel), "WA": sum(c['WA'] for c in sel), "Push": sum(c['Push'] for c in sel), "SMS": sum(c['SMS'] for c in sel), "Email": sum(c['Email'] for c in sel)}
    
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Total Base", f"{sums['Base']:,}"); r2.metric("WhatsApp", f"{sums['WA']:,}"); r3.metric("Mobile Push", f"{sums['Push']:,}"); r4.metric("SMS", f"{sums['SMS']:,}"); r5.metric("Email", f"{sums['Email']:,}")

    f1, f2, f3, f4, f5 = st.columns(5)
    wa_r, sms_r, em_r = f1.number_input("WA ₹", 0.78), f2.number_input("SMS ₹", 0.13), f3.number_input("Email ₹", 0.03)
    conv, aov = f4.slider("Conv %", 0.1, 5.0, 1.0), f5.number_input("AOV ₹", 800)

    def calc(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        roi = (rev/spend) if spend > 0 else 0
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(rev):,}", "ROI": f"{roi:.1f}x"}

    st.table(pd.DataFrame([calc("Mobile Push", sums['Push'], 0.0), calc("WhatsApp", sums['WA'], wa_r), calc("SMS", sums['SMS'], sms_r), calc("Email", sums['Email'], em_r)]))
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run_page()
