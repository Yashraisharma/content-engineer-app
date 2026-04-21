import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai

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
    """Fetches real-time weather using Open-Meteo API."""
    coords = {
        "mumbai": (19.0760, 72.8777), "delhi": (28.6139, 77.2090),
        "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
        "chennai": (13.0827, 80.2707), "kolkata": (22.5726, 88.3639)
    }
    
    if city_key not in coords:
        return fallback_string
        
    lat, lon = coords[city_key]
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code&timezone=Asia%2FKolkata"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        
        if "current" in res:
            current = res["current"]
            temp = current.get("temperature_2m", "--")
            humidity = current.get("relative_humidity_2m", "--")
            code = current.get("weather_code", 0)
            
            if code == 0: condition = "Clear"
            elif code in [1, 2, 3]: condition = "Partly Cloudy"
            elif code in [45, 48]: condition = "Haze"
            elif code in [51, 53, 55, 61, 63, 65, 80, 81, 82]: condition = "Rain"
            elif code in [95, 96, 99]: condition = "Thunderstorm"
            else: condition = "Mist"
            
            return f"{temp}°C | {condition} | Humidity: {humidity}%"
        else:
            return fallback_string
    except Exception:
        return fallback_string

# --- 2. THE REAL-TIME AI GENERATOR (GEMINI 3 FLASH) ---
def generate_live_ai_xsell(sheet_context, target_name, segment_def, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        # Using the correct model string for Gemini 3 Flash Preview
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        You are a clinical retail strategist for Apollo Pharmacy in India.
        Context: 
        - Target City: {city} 
        - Current Weather: {weather} 
        - Business Category: {sheet_context}
        - Target Item/Cohort: {target_name} 
        - Segment Definition: {segment_def}
        
        Generate 5 specific, logical cross-sell product pairs available at an Indian pharmacy. 
        Focus heavily on current weather and segment definition. 
        Respond ONLY with a valid JSON array of arrays:
        [["Anchor Product", "Cross-Sell Product", "Brief strategic reason"]]
        """
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"AI Connection Error: {e}")
        return None

# --- 3. STATIC DEMOGRAPHICS & SEGMENTS ---
DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "31°C | Mist | Humidity: 63%"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "36°C | Heat Alert | Humidity: 21%"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "31°C | Clear | Humidity: 36%"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "35°C | Yellow Alert | Humidity: 35%"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "30°C | Partly Cloudy | Humidity: 79%"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "30°C | Mist | Humidity: 84%"}
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

    # --- DATA LOAD & SKU ROW PARSING ---
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
                    r_current, r_next = df.iloc[i], df.iloc[i+1]
                    if str(r_current.iloc[0]).lower() in ['city', 'category', 'segment', 'nan']: continue
                    
                    primary_id = str(r_current.iloc[0]).strip()
                    secondary_name = str(r_next.iloc[0]).strip()
                    ui_name = f"{secondary_name} (SKU: {primary_id})" if secondary_name.lower() != 'nan' else primary_id
                    ai_name = secondary_name if secondary_name.lower() != 'nan' else primary_id

                    try:
                        rows.append({
                            'UI_Name': ui_name, 'AI_Name': ai_name,
                            'Total': int(r_current.iloc[1]), 'WA': int(r_current.iloc[7]), 
                            'Push': int(r_current.iloc[3]), 'SMS': int(r_current.iloc[4]), 
                            'Email': int(r_current.iloc[5]), 'Sheet': s.title().replace("_", " ")
                        })
                    except: pass
            return rows
        except: return []

    all_rows = get_data()

    # --- ENHANCED BUCKETING ENGINE ---
    city_rows, focus_rows, daily_rows, circle_rows, sku_rows = [], [], [], [], []
    
    # Comprehensive city list including abbreviations and typos
    city_tags = {
        'hyd', 'blr', 'del', 'mum', 'chn', 'kol', 'ncr', 'hyderabad', 'bangalore', 
        'banglore', 'bengaluru', 'mumbai', 'chennai', 'delhi', 'kolkata', 'hydewrabad', 'hyderbad'
    }

    for r in all_rows:
        name_lower = r['UI_Name'].lower()
        sheet_lower = r['Sheet'].lower()
        name_words = set(name_lower.replace('-', ' ').replace('_', ' ').split())
        
        # Priority 1: Circle
        if 'circle' in name_lower or 'circle' in sheet_lower:
            circle_rows.append(r)
        # Priority 2: SKU
        elif 'sku' in sheet_lower or 'sku' in name_lower:
            sku_rows.append(r)
        # Priority 3: City (Checks abbreviations, typos, and full names)
        elif 'city' in sheet_lower or any(tag in name_words for tag in city_tags) or any(c in name_lower for c in DEMOGRAPHICS.keys()):
            city_rows.append(r)
        # Priority 4: Daily Pharma
        elif 'daily' in sheet_lower or 'portfolio' in sheet_lower:
            daily_rows.append(r)
        # Priority 5: Fallback to Focus Category
        else:
            focus_rows.append(r)

    # --- UI: 5 EXPLICIT SELECTIONS ---
    st.markdown("### 📂 Select Target Cohorts")
    selected_cohorts = []
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        opts = {r['UI_Name']: r for r in city_rows}
        p1 = st.multiselect("🏙️ City", options=list(opts.keys()))
        for p in p1: selected_cohorts.append(opts[p])
    with c2:
        opts = {r['UI_Name']: r for r in focus_rows}
        p2 = st.multiselect("🎯 Focus Category", options=list(opts.keys()))
        for p in p2: selected_cohorts.append(opts[p])
    with c3:
        opts = {r['UI_Name']: r for r in daily_rows}
        p3 = st.multiselect("💊 Daily Pharma", options=list(opts.keys()))
        for p in p3: selected_cohorts.append(opts[p])
    with c4:
        opts = {r['UI_Name']: r for r in circle_rows}
        p4 = st.multiselect("⭐ Circle", options=list(opts.keys()))
        for p in p4: selected_cohorts.append(opts[p])
    with c5:
        opts = {r['UI_Name']: r for r in sku_rows}
        p5 = st.multiselect("📦 SKU Based", options=list(opts.keys()))
        for p in p5: selected_cohorts.append(opts[p])

    if not selected_cohorts:
        st.info("👋 Select cohorts above to activate live intelligence.")
        return

    # Determine implicit city (Defaults to Hyderabad)
    implicit_city = "hyderabad"
    for c in selected_cohorts:
        for city in DEMOGRAPHICS.keys():
            if city in c['UI_Name'].lower() or city[:3] in c['UI_Name'].lower():
                implicit_city = city
                break

    # --- ENGINE TABS ---
    st.divider()
    tabs = st.tabs([f"{c['AI_Name'][:20]}..." for c in selected_cohorts])

    def apl_link(display_name):
        url = f"https://www.apollopharmacy.in/search-medicines/{display_name.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {display_name}</a>'

    for i, cohort in enumerate(selected_cohorts):
        with tabs[i]:
            ui_name, ai_name, sheet_origin = cohort['UI_Name'], cohort['AI_Name'], cohort['Sheet']
            city_key = next((c for c in DEMOGRAPHICS.keys() if c in ui_name.lower() or c[:3] in ui_name.lower()), implicit_city)
            dna = DEMOGRAPHICS[city_key]
            seg_key = next((k for k in SEGMENT_DEFS.keys() if k in ui_name.lower()), "active")
            current_seg_def = SEGMENT_DEFS.get(seg_key, 'General Healthcare Cohort')
            
            with st.spinner(f"Syncing Live Context..."):
                common_news = fetch_news(f"{city_key} top headlines")
                health_news = fetch_news(f"{ai_name} healthcare trends India")
                live_weather = fetch_live_weather(city_key, dna["fallback"])

            st.markdown(f"### 🕵️ {ui_name}")
            st.markdown(f"**Category:** {sheet_origin} | **Location:** {city_key.upper()} 🌡️ {live_weather}")
            
            nc1, nc2 = st.columns(2)
            nc1.info(f"**📰 Common News:** [{common_news['title']}]({common_news['link']})")
            nc2.success(f"**🏥 Health News:** [{health_news['title']}]({health_news['link']})")
            
            st.divider()

            # AI Logic
            ai_state_key = f"ai_data_{sheet_origin}_{ui_name}"
            api_key = st.secrets.get("GEMINI_API_KEY")

            if ai_state_key not in st.session_state or st.session_state[ai_state_key] is None:
                if api_key:
                    with st.spinner("🧠 Auto-generating strategy..."):
                        st.session_state[ai_state_key] = generate_live_ai_xsell(sheet_origin, ai_name, current_seg_def, live_weather, city_key, api_key)

            if st.button("🔄 Refresh Strategy", key=f"btn_{sheet_origin}_{ui_name}"):
                st.session_state[ai_state_key] = generate_live_ai_xsell(sheet_origin, ai_name, current_seg_def, live_weather, city_key, api_key)

            active_data = st.session_state.get(ai_state_key)
            if active_data:
                final_rows = [[apl_link(row[0]), apl_link(row[1]), row[2]] for row in active_data]
                df_xsell = pd.DataFrame(final_rows, columns=["User Purchase", "Push (Linked)", "Live AI Reasoning"])
                st.markdown(df_xsell.to_html(escape=False, index=False), unsafe_allow_html=True)

    # --- ROI FORECAST ---
    st.divider()
    t_base = sum([c['Total'] for c in selected_cohorts])
    t_wa = sum([c['WA'] for c in selected_cohorts])
    t_push = sum([c['Push'] for c in selected_cohorts])
    t_sms = sum([c['SMS'] for c in selected_cohorts])
    t_email = sum([c['Email'] for c in selected_cohorts])

    st.subheader("🧬 Aggregated ROI Forecast")
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
