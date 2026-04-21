import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
import json
import google.generativeai as genai

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
        url = f"https://wttr.in/{city}?format=%t+|+%C+|+Humidity:+%h&m"
        res = requests.get(url, timeout=5)
        return f"🌡️ {res.text.strip()}" if res.status_code == 200 else "🌡️ Live Weather Syncing..."
    except: return "🌡️ Weather Service Offline"

# --- 2. THE REAL-TIME AI GENERATOR ---
def generate_live_ai_xsell(api_key, category, segment_def, weather, city):
    """Calls Gemini API to generate real-time, logical cross-sells on the fly."""
    if not api_key:
        return None # Falls back to pre-set data if no key is provided
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a clinical retail strategist for Apollo Pharmacy in India.
        Context:
        - Target City: {city}
        - Current Weather: {weather}
        - Target Category: {category}
        - User Segment: {segment_def}
        
        Generate 5 highly specific, logical cross-sell pairs available at an Indian pharmacy. 
        If the segment is 'Churn/Winback', focus on heavy discounts/hero items. If 'Power', focus on bulk/subscriptions.
        Ensure products match the weather (e.g., electrolytes/sunscreen for heat, immunity for monsoon).
        
        Respond ONLY with a valid JSON array of arrays in this exact format, with no markdown formatting:
        [
            ["Anchor Product Name", "Logical Cross-Sell Product", "Brief 1-sentence strategic reason why this works now"]
        ]
        """
        response = model.generate_content(prompt)
        # Clean the response to ensure it parses correctly
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        st.error(f"AI Generation Failed: {e}")
        return None

# --- 3. DEMOGRAPHICS & SEGMENTS ---
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
    "enhancement": "High-volume users with many transactions"
}

def run_page():
    now = datetime.now()
    st.set_page_config(layout="wide")
    st.header("🛡️ Strategic Growth Predictor")
    
    # --- API KEY INPUT ---
    st.sidebar.title("🧠 AI Engine")
    st.sidebar.markdown("Enter a Google Gemini API Key to enable real-time, dynamic cross-sell generation.")
    api_key = st.sidebar.text_input("Gemini API Key:", type="password")

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
            city_key = next((c for c in DEMOGRAPHICS.keys() if c in p_lower), "hyderabad")
            dna = DEMOGRAPHICS[city_key]
            seg_key = next((k for k in SEGMENT_DEFS.keys() if k in p_lower), "active")
            current_seg_def = SEGMENT_DEFS.get(seg_key, 'General Healthcare Cohort')
            
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
                            <span style="background: #e2e8f0; color: #334155; padding: 4px 12px; border-radius: 15px; font-size: 0.85em; font-weight: 600;">📖 {current_seg_def}</span>
                        </div>
                        <span style="background: #ef4444; color:#fff; padding: 5px 15px; border-radius: 20px; font-weight: bold;">{live_weather} | {city_key.upper()}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- LIVE AI GENERATION TRIGGER ---
            st.subheader("🛒 Real-Time Strategy Matrix")
            
            # Use session state to hold the AI generated data so it doesn't disappear
            ai_state_key = f"ai_data_{primary}"
            if ai_state_key not in st.session_state:
                st.session_state[ai_state_key] = None

            c1, c2 = st.columns([1, 4])
            with c1:
                if st.button("🧠 Generate Live AI Strategy", key=f"btn_{primary}"):
                    if not api_key:
                        st.warning("⚠️ Please enter an API key in the sidebar to generate live strategies.")
                    else:
                        with st.spinner("AI is analyzing weather, city, and segment..."):
                            st.session_state[ai_state_key] = generate_live_ai_xsell(api_key, primary, current_seg_def, live_weather, city_key)

            # --- RENDER THE TABLE ---
            active_data = st.session_state[ai_state_key]
            
            if active_data:
                # If AI successfully generated data, format it with links
                final_rows = [[apl_link(row[0]), apl_link(row[1]), row[2]] for row in active_data]
                df_xsell = pd.DataFrame(final_rows, columns=["User Purchase", "Push (Linked)", "Live AI Reasoning"])
                st.success("✅ AI Strategy Generated Successfully based on current context.")
                st.markdown(df_xsell.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info("👆 Click 'Generate Live AI Strategy' to ping the LLM and create 5 unique combinations for this segment right now.")

            st.write("")

    # --- AGGREGATED ROI FORECAST ---
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
