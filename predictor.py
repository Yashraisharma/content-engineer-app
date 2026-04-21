import streamlit as st
import pandas as pd
from datetime import datetime

def run_page():
    now = datetime.now()
    current_date = now.strftime("%A, %d %B %Y")
    
    st.header("📊 Segment Analysis & Reach Predictor")
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def get_data():
        try:
            sheets = ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
            rows = []
            for s in sheets:
                df = pd.read_excel(EXCEL_URL, sheet_name=s, engine='openpyxl').dropna(how='all').reset_index(drop=True)
                for i in range(0, len(df), 2):
                    r = df.iloc[i]
                    if str(r.iloc[0]).lower() in ['city', 'category', 'segment', 'status']: continue
                    rows.append({
                        'Name': str(r.iloc[0]).strip(), 
                        'Total': int(r.iloc[1]) if pd.notna(r.iloc[1]) else 0, 
                        'WA': int(r.iloc[7]) if pd.notna(r.iloc[7]) else 0, 
                        'Push': int(r.iloc[3]) if pd.notna(r.iloc[3]) else 0, 
                        'SMS': int(r.iloc[4]) if pd.notna(r.iloc[4]) else 0, 
                        'Email': int(r.iloc[5]) if pd.notna(r.iloc[5]) else 0
                    })
            return pd.DataFrame(rows)
        except: return pd.DataFrame()

    df_master = get_data()

    # --- 1. BETTER SELECTION PROCESS ---
    st.sidebar.markdown("### 🎯 Segment Manager")
    if st.sidebar.button("🗑️ Clear All Selections"):
        st.session_state.selected_segments = []
        st.rerun()

    # Improved selection: Multiselect with default from Code X
    picks = st.multiselect(
        "🔍 Search & Add Segments to Analysis:", 
        options=df_master['Name'].unique().tolist(),
        default=st.session_state.get("selected_segments", [])
    )
    # Sync back to session state so Code X knows what we are analyzing here
    st.session_state.selected_segments = picks

    if not picks:
        st.info("👋 Select one or more segments above to begin real-time analysis.")
        return

    # --- 2. DYNAMIC CROSS-SELL ENGINE (FIXED) ---
    st.divider()
    primary_segment = picks[0] # Focus intelligence on the first selected segment
    
    # Categorization Logic for Products
    is_chronic = any(x in primary_segment for x in ["Cardio", "Diab", "Pharma", "Chronic"])
    is_metro = any(x in primary_segment for x in ["Hyderabad", "Delhi", "Bangalore", "Mumbai"])
    is_motherhood = "mom" in primary_segment.lower() or "baby" in primary_segment.lower()

    if is_chronic:
        p1 = {"name": "Apollo Pharmacy BP Monitor", "link": "https://www.apollopharmacy.in/shop-by-category/health-devices"}
        p2 = {"name": "Apollo Life Sugar-Free Protein", "link": "https://www.apollopharmacy.in/shop-by-category/diabetes-care"}
        intel = {"old": "65%", "moms": "2%", "tech": "45%", "type": "Chronic/Patient"}
    elif is_motherhood:
        p1 = {"name": "Apollo Pharmacy Baby Diapers (Value Pack)", "link": "https://www.apollopharmacy.in/shop-by-category/baby-care"}
        p2 = {"name": "Apollo Life Pregnancy Supplement", "link": "https://www.apollopharmacy.in/shop-by-category/women-care"}
        intel = {"old": "5%", "moms": "95%", "tech": "92%", "type": "Motherhood/New Parent"}
    elif is_metro:
        p1 = {"name": "ORSL Rehydrate Apple (Heatwave Essential)", "link": "https://www.apollopharmacy.in/shop-by-category/otc"}
        p2 = {"name": "Apollo Pharmacy SPF 50 Sunscreen", "link": "https://www.apollopharmacy.in/shop-by-category/apollo-personal-care"}
        intel = {"old": "15%", "moms": "8%", "tech": "94%", "type": "Urban/Metropolitan"}
    else:
        p1 = {"name": "Apollo Life Multivitamin Gummies", "link": "https://www.apollopharmacy.in/shop-by-category/vitamins-and-supplements"}
        p2 = {"name": "Apollo Pharmacy First Aid Kit", "link": "https://www.apollopharmacy.in/shop-by-category/otc"}
        intel = {"old": "18%", "moms": "5%", "tech": "70%", "type": "General Consumer"}

    # --- 3. DISPLAY INTELLIGENCE & RECOMMENDATIONS ---
    st.markdown(f"""
        <div style="background-color: #f0f9ff; border-left: 5px solid #0ea5e9; padding: 15px; border-radius: 5px;">
            <p style="margin: 0; color: #0c4a6e;"><b>🕒 Live Intel ({current_date}): {primary_segment}</b></p>
            <div style="display: flex; justify-content: space-between; font-size: 0.85em; margin-top: 10px;">
                <span>👴 Seniors: <b>{intel['old']}</b></span>
                <span>🍼 Moms: <b>{intel['moms']}</b></span>
                <span>📱 Tech: <b>{intel['tech']}</b></span>
                <span>🏷️ Class: <b>{intel['type']}</b></span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛒 Targeted Cross-Sell Recommendations")
    cs1, cs2 = st.columns(2)
    cs1.info(f"**Primary Push:**\n{p1['name']}")
    cs1.markdown(f"[View Product]({p1['link']})")
    cs2.success(f"**Logical Upsell:**\n{p2['name']}")
    cs2.markdown(f"[View Product]({p2['link']})")

    # --- 4. DATA AGGREGATION & ROI (REMAINS UNCHANGED) ---
    st.divider()
    combined_data = df_master[df_master['Name'].isin(picks)].sum(numeric_only=True)
    st.subheader("🧬 Reach DNA (Combined)")
    # ... [Rest of your ROI Table and Strategy Verdict code here] ...
