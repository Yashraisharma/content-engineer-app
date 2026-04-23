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
    return {"title": "Market intelligence currently offline", "link": "#"}

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
    Strategic Engine: Focuses on high-margin FMCG, Baby, and Wellness.
    Uses versioned model cycling to avoid 404 errors.
    """
    if not api_key: return None
    genai.configure(api_key=api_key)
    
    # Cycling through model aliases to ensure 2026 endpoint compatibility
    model_names = ['gemini-3-flash', 'gemini-1.5-flash', 'gemini-2.0-flash']
    
    prompt = f"""
    Role: Clinical Growth Strategist for Apollo Pharmacy India.
    Anchor Product: {name} (Category: {sheet})
    Target Segment: {segment} | Environment: {city}, {weather}

    TASK: Suggest 5 logical growth pairings.
    STRATEGIC RULES:
    1. PRIORITIZE: High-margin FMCG, Baby Care (Diapers/Wipes), Hygiene (Antifungal towels), Health Devices, and Clinical Nutrition.
    2. NO RX MEDS: Do not suggest prescription drugs.
    3. LOGIC: Focus on Moisture Management, Skin Barrier Repair, and internal wellness.
    
    Format: Respond ONLY with a JSON array of arrays: [["Anchor", "Cross-Sell", "Reason"]]
    """
    
    for m_name in model_names:
        try:
            model = genai.GenerativeModel(m_name)
            response = model.generate_content(prompt)
            match = re.search(r'\[\s*\[.*\]\s*\]', response.text, re.DOTALL)
            if match: return json.loads(match.group())
        except: continue
    return [["System Pause", "AI Unreachable", "Please check your API key or model availability."]]

# --- 2. CONFIG & THEMES ---

DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "31°C | Mist"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "36°C | Heat Alert"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "31°C | Clear"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "35°C | Yellow Alert"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "30°C | Cloudy"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "30°C | Mist"}
}

def apply_custom_theme(mode):
    if mode == "Midnight (Dark)":
        bg, text, card, accent = "#0f172a", "#f1f5f9", "#1e293b", "#60a5fa"
    else:
        bg, text, card, accent = "#fcfcfc", "#1e293b", "#ffffff", "#2563eb"
    
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {bg}; }}
        h1, h2, h3, h4, p, span, label, div {{ color: {text} !important; }}
        [data-testid="stMetricLabel"] {{ color: {text} !important; font-weight: 700 !important; font-size: 1rem; opacity: 0.85; }}
        [data-testid="stMetricValue"] {{ color: {text} !important; font-weight: 800 !important; }}
        div[data-testid="stMetric"] {{ background-color: {card}; border: 1px solid {accent}33; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .news-box {{ background-color: {card}; border-left: 5px solid {accent}; padding: 15px; border-radius: 8px; margin-bottom: 12px; }}
        a {{ color: {accent} !important; font-weight: 700; text-decoration: none; }}
        .roi-panel {{ background-color: {card}; padding: 30px; border-radius: 20px; border: 1.5px solid {accent}; margin-top: 25px; }}
        table {{ color: {text} !important; background-color: {card}; width: 100%; border-radius: 10px; overflow: hidden; }}
        th {{ background-color: {accent}11; padding: 12px; border-bottom: 2px solid {accent}33; }}
        td {{ padding: 12px; border-bottom: 1px solid {accent}11; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. DASHBOARD ENGINE ---

def run_dashboard():
    st.set_page_config(page_title="Apollo Growth Strategist", layout="wide")
    
    with st.sidebar:
        st.image("https://www.apollopharmacy.in/static/images/logo.svg", width=160)
        st.divider()
        theme_choice = st.radio("Display Mode:", ["Professional (Light)", "Midnight (Dark)"])
        st.divider()
        st.caption("Strategic Growth Engine v4.0 (2026 Build)")

    apply_custom_theme(theme_choice)
    now = datetime.now()
    st.title("🛡️ Strategic Growth Predictor")
    st.markdown(f"**Enterprise Sync:** {now.strftime('%A, %d %B | %I:%M %p')}")

    # --- DATA PARSER ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"
    
    @st.cache_data
    def load_cohort_data():
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
                    
                    # Logic: ID on Row 1, Name on Row 2
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

    all_cohorts = load_cohort_data()

    # --- SEGMENTATION LOGIC ---
    city_group, focus_group, portfolio_group, circle_group, sku_group = [], [], [], [], []
    # Comprehensive Regex for Cities + Common Typos
    city_regex = re.compile(r'\b(hyd|hyderabad|hyderbad|hydewrabad|blr|bangalore|banglore|del|delhi|mum|mumbai|chn|chennai|kol|kolkata|ncr|bengaluru)\b', re.I)

    for c in all_cohorts:
        name_clean, sheet_clean = c['UI_Name'].lower(), c['Source_Sheet'].lower()
        if 'circle' in name_clean: circle_group.append(c)
        elif 'top 6 cities' in sheet_clean or city_regex.search(name_clean): city_group.append(c)
        elif 'pharma_focus' in sheet_clean: focus_group.append(c)
        elif 'daily_pharma' in sheet_clean: portfolio_group.append(c)
        else: sku_group.append(c)

    # --- SELECTION HUB ---
    st.markdown("### 📂 Selection Hub")
    cols = st.columns(5)
    active_selection = []
    
    with cols[0]:
        p = st.multiselect("🏙️ Cities", options=[x['UI_Name'] for x in city_group])
        for val in p: active_selection.append(next(x for x in city_group if x['UI_Name'] == val))
    with cols[1]:
        p = st.multiselect("🎯 Focus", options=[x['UI_Name'] for x in focus_group])
        for val in p: active_selection.append(next(x for x in focus_group if x['UI_Name'] == val))
    with cols[2]:
        p = st.multiselect("💊 Portfolio", options=[x['UI_Name'] for x in portfolio_group])
        for val in p: active_selection.append(next(x for x in portfolio_group if x['UI_Name'] == val))
    with cols[3]:
        p = st.multiselect("⭐ Circle", options=[x['UI_Name'] for x in circle_group])
        for x in p: active_selection.append(next(y for y in circle_group if y['UI_Name'] == x))
    with cols[4]:
        p = st.multiselect("📦 SKU", options=[x['UI_Name'] for x in sku_group])
        for x in p: active_selection.append(next(y for y in sku_group if y['UI_Name'] == x))

    if not active_selection:
        st.info("👋 Use the selection blocks above to start your growth analysis.")
        return

    # --- GROWTH TABS ---
    st.divider()
    growth_tabs = st.tabs([f"📈 {c['AI_Name'][:18]}" for c in active_selection])

    for i, cohort in enumerate(active_selection):
        with growth_tabs[i]:
            # Geographic Routing
            city_found = "hyderabad"
            for k in DEMOGRAPHICS.keys():
                if k[:3] in cohort['UI_Name'].lower() or k in cohort['UI_Name'].lower(): 
                    city_found = k; break
            
            dna = DEMOGRAPHICS[city_found]
            live_weather = fetch_live_weather(city_found, dna['fallback'])

            # Live Context Cards
            c_left, c_right = st.columns(2)
            c_left.markdown(f'<div class="news-box"><b>🌐 Local Trends:</b> {fetch_news(city_found)["title"]}</div>', unsafe_allow_html=True)
            c_right.markdown(f'<div class="news-box"><b>🏥 Wellness Pulse:</b> {fetch_news(cohort["AI_Name"])["title"]}</div>', unsafe_allow_html=True)

            # Core Metrics
            st.markdown(f"#### 🧬 {city_found.upper()} Context | {live_weather}")
            m_cols = st.columns(4)
            m_cols[0].metric("👵 Seniors", dna['seniors'])
            m_cols[1].metric("🍼 Moms", dna['moms'])
            m_cols[2].metric("👩 Female", dna['females'])
            m_cols[3].metric("📱 Tech Savvy", dna['tech'])

            # Strategic AI Synthesis
            st.divider()
            st.subheader("🛒 Strategic Cross-Sell Synthesis (High-Margin Focus)")
            
            state_key = f"strat_v5_{cohort['UI_Name']}"
            if state_key not in st.session_state or st.session_state[state_key] is None:
                with st.spinner("Flash 3 analyzing clinical wellness drivers..."):
                    st.session_state[state_key] = generate_growth_strategy(cohort['Source_Sheet'], cohort['AI_Name'], "Standard", live_weather, city_found, st.secrets["GEMINI_API_KEY"])

            if st.session_state[state_key]:
                def link(v): return f'<a href="https://www.apollopharmacy.in/search-medicines/{v.replace(" ","%20")}" target="_blank">🛒 {v}</a>'
                df_view = pd.DataFrame([[link(r[0]), link(r[1]), r[2]] for r in st.session_state[state_key]], columns=["Anchor", "Strategic Pairing", "Growth Logic"])
                st.markdown(df_view.to_html(escape=False, index=False), unsafe_allow_html=True)
                st.button("🔄 Refresh Strategic Pairs", key=f"re_btn_{i}")

    # --- ROI PERFORMANCE PANEL ---
    st.markdown('<div class="roi-panel">', unsafe_allow_html=True)
    st.subheader("🧬 Aggregated Campaign Impact")
    
    sums = {
        "Base": sum(c['Total'] for c in active_selection),
        "WA": sum(c['WA'] for c in active_selection),
        "Push": sum(c['Push'] for c in active_selection),
        "SMS": sum(c['SMS'] for c in active_selection),
        "Email": sum(c['Email'] for c in active_selection)
    }
    
    r_cols = st.columns(5)
    r_cols[0].metric("Total Base", f"{sums['Base']:,}")
    r_cols[1].metric("WhatsApp", f"{sums['WA']:,}")
    r_cols[2].metric("Mobile Push", f"{sums['Push']:,}")
    r_cols[3].metric("SMS", f"{sums['SMS']:,}")
    r_cols[4].metric("Email", f"{sums['Email']:,}")

    st.markdown("#### ⚙️ Financial Simulation")
    f_cols = st.columns(5)
    cost_wa = f_cols[0].number_input("WA ₹", 0.78)
    cost_sms = f_cols[1].number_input("SMS ₹", 0.13)
    cost_em = f_cols[2].number_input("Email ₹", 0.03)
    p_conv = f_cols[3].slider("Conv %", 0.1, 5.0, 1.0)
    p_aov = f_cols[4].number_input("AOV ₹", 800)

    def calculate_roi(name, reach, unit_cost):
        revenue = (reach * (p_conv/100)) * p_aov
        spend = reach * unit_cost
        roi_val = (revenue/spend) if spend > 0 else 0
        return {"Channel": name, "Reach": f"{int(reach):,}", "Spend": f"₹{int(spend):,}", "Revenue": f"₹{int(revenue):,}", "ROI": f"{roi_val:.1f}x"}

    results = [
        calculate_roi("Mobile Push (Free)", sums['Push'], 0.0),
        calculate_roi("WhatsApp Business", sums['WA'], cost_wa),
        calculate_roi("Transactional SMS", sums['SMS'], cost_sms),
        calculate_roi("Email Marketing", sums['Email'], cost_em)
    ]
    st.table(pd.DataFrame(results))
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run_dashboard()
