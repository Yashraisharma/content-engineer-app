import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# --- 1. LIVE GOOGLE NEWS API INTEGRATION ---
@st.cache_data(ttl=900)
def fetch_google_news(query, count=1):
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=5)
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('./channel/item')[:count]:
            title = item.find('title').text.split(' - ')[0]
            link = item.find('link').text
            news_items.append({"title": title, "link": link})
        return news_items
    except Exception as e:
        return [{"title": f"Live feed temporarily unavailable for: {query}", "link": "#"}]

def run_page():
    # --- 2. CORE CONFIG ---
    now = datetime.now()
    st.header("🛡️ Live Growth Command Center")
    st.markdown(f"**System Sync:** {now.strftime('%A, %d %B %Y | %I:%M %p')} | **Engine:** Live API")

    # --- 3. EXCEL DATA INTEGRATION ---
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def get_excel_data():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            rows = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all').reset_index(drop=True)
                for i in range(0, len(df), 2):
                    r = df.iloc[i]
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment']: continue
                    rows.append({
                        'Name': str(r.iloc[0]).strip(), 
                        'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 
                        'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0,
                        'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0
                    })
            return pd.DataFrame(rows)
        except Exception as e: 
            return pd.DataFrame()

    df_master = get_excel_data()

    # --- 4. ADVANCED MULTI-SELECT ---
    if "selected_segments" not in st.session_state:
        st.session_state.selected_segments = []

    def sync_picks():
        st.session_state.selected_segments = st.session_state.ms_key

    st.sidebar.title("🎮 Cohort Targeting")
    is_circle = st.sidebar.checkbox("🟢 Target CIRCLE Subscribers Only")
    
    picks = st.multiselect(
        "🔍 Search & Analyze Live Segments (Select Multiple):", 
        options=df_master['Name'].unique().tolist() if not df_master.empty else [],
        default=st.session_state.selected_segments,
        key="ms_key",
        on_change=sync_picks
    )

    if not picks:
        st.info("👋 Select one or more target cohorts above to pull live Google data.")
        return

    # --- 5. THE LIVE INTELLIGENCE LOOP ---
    st.divider()
    st.subheader("📡 Live Context & Strategic Action")
    
    tabs = st.tabs([p for p in picks])

    for i, primary in enumerate(picks):
        with tabs[i]:
            primary_lower = primary.lower()
            
            # --- ENTITY EXTRACTION ---
            cities = ["mumbai", "delhi", "bangalore", "hyderabad", "chennai", "kolkata"]
            active_city = next((c for c in cities if c in primary_lower), "India")
            
            categories = {"cardio": "Heart Health", "diab": "Diabetes", "mom": "Maternal", "baby": "Pediatric", "skin": "Dermatology"}
            active_category = next((v for k, v in categories.items() if k in primary_lower), "General Healthcare")
            
            segments = ["ntu", "churn", "winback", "active", "power", "enhancement", "new registered"]
            active_segment = next((s for s in segments if s in primary_lower), "active")

            # --- API EXECUTION: GOOGLE NEWS ---
            with st.spinner(f"Pulling live Google data for {active_city}..."):
                gen_news = fetch_google_news(f"{active_city} top local news headlines today", count=1)[0]
                health_news = fetch_google_news(f"{active_category} health news India OR {active_city} healthcare", count=1)[0]

            # --- BASKET AFFINITY LOGIC ---
            if "mom" in primary_lower or "baby" in primary_lower:
                demographics = "👶 Pediatric | 🍼 95% Moms | 👵 2% Seniors"
                p1, p1_url = "Pampers Baby-Dry Diapers", "https://www.apollopharmacy.in/shop-by-category/baby-care/diapers"
                p2, p2_url = "Himalaya Gentle Baby Wipes", "https://www.apollopharmacy.in/shop-by-category/baby-care/baby-wipes"
            elif "cardio" in primary_lower or "diab" in primary_lower:
                demographics = "💊 Chronic Care | 🍼 1% Moms | 👵 82% Seniors"
                p1, p1_url = "Apollo Pharmacy Digital BP Monitor", "https://www.apollopharmacy.in/shop-by-category/health-devices/bp-monitors"
                p2, p2_url = "Apollo Life Sugar-Free Protein", "https://www.apollopharmacy.in/shop-by-category/diabetes-care"
            elif "skin" in primary_lower:
                demographics = "🧴 Derma Care | 🙋‍♀️ 60% Gen-Z/Millennial"
                p1, p1_url = "Cetaphil Gentle Skin Cleanser", "https://www.apollopharmacy.in/shop-by-category/skin-care"
                p2, p2_url = "Apollo SPF 50 Sunscreen", "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care/sun-care"
            else:
                demographics = "🏢 General Urban | Mixed Demographic"
                p1, p1_url = "ORSL Electrolyte Drink", "https://www.apollopharmacy.in/shop-by-category/otc"
                p2, p2_url = "Apollo Life Multivitamins", "https://www.apollopharmacy.in/shop-by-category/vitamins-and-supplements"

            # Behavioral Overrides
            pitch_tone = "Standard Restock"
            if active_segment in ["churn", "winback"]:
                pitch_tone = "🚨 HIGH PRIORITY WINBACK: Offer 25% Off + Free Delivery to reactivate."
            elif active_segment == "ntu" or active_segment == "new registered":
                pitch_tone = "👋 FIRST TIME USER (NTU): Focus on Apollo Trust, Quality, and 15% First-Order Welcome coupon."
            elif active_segment in ["power", "active", "enhancement"]:
                if is_circle:
                    pitch_tone = "👑 CIRCLE POWER USER: Upsell bulk packs (Monthly Supplies) and push 2-hour rapid delivery."
                else:
                    pitch_tone = "⭐ ACTIVE NON-CIRCLE: Perfect target to upsell CIRCLE Membership with their next order."

            # --- THE UI RENDER ---
            st.markdown(f"""
                <div style="background-color: #f8fafc; border: 2px solid #334155; padding: 25px; border-radius: 15px; margin-bottom: 20px; color: #000;">
                    <h3 style="margin-top: 0; color: #0f172a;">📍 {primary.upper()} ({active_city.capitalize()})</h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div style="background: white; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6;">
                            <b>📰 Popular Local News (Live):</b><br>
                            <a href="{gen_news['link']}" target="_blank" style="color: #1d4ed8; text-decoration: none;">{gen_news['title']}</a>
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 8px; border-left: 5px solid #10b981;">
                            <b>🏥 {active_category} Health News (Live):</b><br>
                            <a href="{health_news['link']}" target="_blank" style="color: #047857; text-decoration: none;">{health_news['title']}</a>
                        </div>
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #e2e8f0; border-radius: 5px;">
                        <b>🧬 Profile:</b> {demographics} | <b>Behavior:</b> {active_segment.upper()}
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Strategy Block
            c1, c2, c3 = st.columns([1, 1, 1.5])
            c1.info(f"**Primary Push:**\n[{p1}]({p1_url})")
            c2.success(f"**Logical Upsell:**\n[{p2}]({p2_url})")
            c3.warning(f"**🧠 AI Pitch Strategy:**\n{pitch_tone}")

    # --- 6. REACH DNA & ROI MATH ---
    st.divider()
    stats = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader(f"🧬 Aggregated Reach DNA & ROI ({len(picks)} Segments Selected)")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Base", f"{int(stats['Total']):,}")
    m2.metric("WhatsApp", f"{int(stats['WA']):,}")
    m3.metric("Mobile Push", f"{int(stats['Push']):,}")
    m4.metric("SMS", f"{int(stats['SMS']):,}")
    m5.metric("Email", f"{int(stats.get('Email', 0)):,}")
    
    st.divider()
    col_v1, col_v2, col_v3 = st.columns(3)
    wa_rate = col_v1.number_input("WA Cost (Karix)", value=0.78)
    sms_rate = col_v2.number_input("SMS Cost (Vi)", value=0.13)
    email_rate = col_v3.number_input("Email Cost", value=0.03)
    
    f1, f2 = st.columns(2)
    conv = f1.slider("Conversion Rate (%)", 0.1, 5.0, 1.0)
    aov = f2.number_input("Average Order Value (₹)", value=800)

    def calc_channel(name, reach, cost):
        rev = (reach * (conv/100)) * aov
        spend = reach * cost
        return {
            "Channel": name, 
            "Reach": f"{int(reach):,}", 
            "Spend": f"₹{int(spend):,}", 
            "Rev": f"₹{int(rev):,}", 
            "ROI": f"{(rev/spend):.1f}x" if spend > 0 else "∞"
        }

    table = [
        calc_channel("Mobile Push", stats['Push'], 0.0),
        calc_channel("WhatsApp", stats['WA'], wa_rate),
        calc_channel("SMS", stats['SMS'], sms_rate),
        calc_channel("Email", stats.get('Email', 0), email_rate)
    ]
    st.table(pd.DataFrame(table))
