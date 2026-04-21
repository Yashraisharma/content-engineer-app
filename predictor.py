import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai

# --- 1. LIVE UTILITIES (NEWS & ENTERPRISE WEATHER) ---
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
    """Fetches highly accurate, real-time weather using Open-Meteo API locked to IST."""
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
            
            return f"🌡️ {temp}°C | {condition} | Humidity: {humidity}%"
        else:
            return fallback_string
    except Exception:
        return fallback_string

# --- 2. THE REAL-TIME AI GENERATOR (GEMINI 3 FLASH) ---
def generate_live_ai_xsell(category, segment_def, weather, city, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        You are a clinical retail strategist for Apollo Pharmacy in India.
        Context: Target City: {city} | Current Weather: {weather} | Category: {category} | Segment: {segment_def}
        
        Generate 5 specific, logical cross-sell product pairs available at an Indian pharmacy. 
        Focus heavily on the weather and the segment definition. 
        Provide UNIQUE combinations not given previously.
        Respond ONLY with a valid JSON array of arrays:
        [["Anchor Product Name", "Logical Cross-Sell Product", "Brief 1-sentence strategic reason"]]
        """
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"AI Connection Error: {e}")
        return None

# --- 3. STATIC DEMOGRAPHICS & SEGMENTS ---
DEMOGRAPHICS = {
    "mumbai": {"seniors": "14.8%", "females": "46.1%", "moms": "12.4%", "tech": "92%", "fallback": "🌡️ 31°C | Mist | Humidity: 63%"},
    "delhi": {"seniors": "12.2%", "females": "46.5%", "moms": "13.8%", "tech": "91%", "fallback": "🌡️ 36°C | Heat Alert | Humidity: 21%"},
    "bangalore": {"seniors": "11.5%", "females": "47.9%", "moms": "12.1%", "tech": "96%", "fallback": "🌡️ 31°C | Clear | Humidity: 36%"},
    "hyderabad": {"seniors": "10.9%", "females": "48.8%", "moms": "11.9%", "tech": "94%", "fallback": "🌡️ 35°C | Yellow Alert | Humidity: 35%"},
    "chennai": {"seniors": "15.2%", "females": "49.7%", "moms": "10.5%", "tech": "90%", "fallback": "🌡️ 30°C | Partly Cloudy | Humidity: 79%"},
    "kolkata": {"seniors": "16.1%", "females": "47.5%", "moms": "11.2%", "tech": "86%", "fallback": "🌡️ 30°C | Mist | Humidity: 84%"}
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

    # --- DATA LOAD ---
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

    # --- TOP CONTROLS ---
    c1, c2 = st.columns([1, 3])
    with c1:
        global_city = st.selectbox("🌍 Base City Context", [k.title() for k in DEMOGRAPHICS.keys()], index=3).lower()
    with c2:
        picks = st.multiselect("🔍 Select Target Cohorts:", options=df_master['Name'].unique().tolist() if not df_master.empty else [], default=st.session_state.selected_segments, key="ms_key", on_change=sync_picks)

    if not picks:
        st.info("👋 Select cohorts above to activate live intelligence.")
        return

    # --- ENGINE TABS ---
    st.divider()
    tabs = st.tabs([p for p in picks])

    def apl_link(display_name):
        url = f"https://www.apollopharmacy.in/search-medicines/{display_name.replace(' ', '%20')}"
        return f'<a href="{url}" target="_blank" style="color: #1d4ed8; font-weight: 600;">🛒 {display_name}</a>'

    for i, primary in enumerate(picks):
        with tabs[i]:
            p_lower = primary.lower()
            city_key = next((c for c in DEMOGRAPHICS.keys() if c in p_lower), global_city)
            dna = DEMOGRAPHICS[city_key]
            seg_key = next((k for k in SEGMENT_DEFS.keys() if k in p_lower), "active")
            current_seg_def = SEGMENT_DEFS.get(seg_key, 'General Healthcare Cohort')
            
            with st.spinner(f"Syncing Live Context for {city_key.title()}..."):
                common_news = fetch_news(f"{city_key} top headlines")
                health_news = fetch_news(f"{primary} healthcare trends India")
                live_weather = fetch_live_weather(city_key, dna["fallback"])

            # --- NATIVE STREAMLIT UI ---
            st.markdown(f"### 🕵️ {primary.upper()}")
            st.markdown(f"**Segment:** 📖 {current_seg_def} | **Location:** {city_key.upper()} ({live_weather})")
            
            # Live News Block
            st.write("#### 📡 Live Intelligence")
            nc1, nc2 = st.columns(2)
            nc1.info(f"**📰 Common News:** [{common_news['title']}]({common_news['link']})")
            nc2.success(f"**🏥 Health News:** [{health_news['title']}]({health_news['link']})")
            
            st.divider()

            # --- AUTONOMOUS AI GENERATION LOGIC ---
            st.subheader("🛒 Real-Time AI Strategy Matrix")
            
            ai_state_key = f"ai_data_{primary}"
            
            # Fetch API Key securely
            try:
                active_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                active_key = None

            # 1. AUTO-RUN ON SELECTION (If no data exists yet for this tab)
            if ai_state_key not in st.session_state or st.session_state[ai_state_key] is None:
                if not active_key:
                    st.error("⚠️ Missing API Key! Check .streamlit/secrets.toml.")
                else:
                    with st.spinner(f"🧠 Auto-generating initial strategy for {primary}..."):
                        st.session_state[ai_state_key] = generate_live_ai_xsell(primary, current_seg_def, live_weather, city_key, active_key)

            # 2. THE REFRESH BUTTON (Placed BEFORE table rendering so it updates instantly)
            c_btn1, c_btn2 = st.columns([1, 4])
            with c_btn1:
                if st.button("🔄 Refresh (Get 5 New Options)", key=f"btn_{primary}"):
                    if active_key:
                        with st.spinner("🧠 Brainstorming new product associations..."):
                            # Overwrites the old data with a fresh API call
                            st.session_state[ai_state_key] = generate_live_ai_xsell(primary, current_seg_def, live_weather, city_key, active_key)

            # 3. RENDER THE TABLE
            active_data = st.session_state.get(ai_state_key)
            if active_data:
                final_rows = [[apl_link(row[0]), apl_link(row[1]), row[2]] for row in active_data]
                df_xsell = pd.DataFrame(final_rows, columns=["User Purchase", "Push (Linked)", "Live AI Reasoning"])
                st.success("✅ AI Strategy Ready.")
                st.markdown(df_xsell.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("No strategy generated. Click refresh to try again.")

            st.write("")

    # --- AGGREGATED ROI FORECAST ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Aggregated Reach & ROI")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    m5.metric("Email", f"{int(stats.get('Email', 0)):,}")

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

    st.table(pd.DataFrame([
        calc("Push", stats['Push'], 0.0), 
        calc("WhatsApp", stats['WA'], wa_rate), 
        calc("SMS", stats['SMS'], sms_rate), 
        calc("Email", stats.get('Email', 0), email_rate)
    ]))

if __name__ == "__main__":
    st.set_page_config(page_title="Strategic Growth Predictor", layout="wide")
    run_page()
