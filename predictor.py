import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown("#### Logic Y: Volume-Based Strategic Planning")
    
    # 1. DATA SOURCE (Points to your Master Excel on GitHub)
    EXCEL_URL = "https://raw.githubusercontent.com/Yashraisharma/content-engineer-app/main/cohort_sheets.xlsx"

    @st.cache_data
    def load_and_stitch_volume(sheet_name):
        try:
            # Loading the raw sheet
            df = pd.read_excel(EXCEL_URL, sheet_name=sheet_name).dropna(how='all').reset_index(drop=True)
            clean_data = []
            
            # Universal 2-row "Stitch" Logic for volumes
            for i in range(0, len(df), 2):
                if i+1 >= len(df): break
                row_names = df.iloc[i]
                
                # Skip header rows
                if str(row_names.iloc[0]).lower() in ['city', 'category', 'segment', 'status']:
                    continue
                
                clean_data.append({
                    'Name': str(row_names.iloc[0]).strip(),
                    'Total_Audience': int(row_names.iloc[1]) if pd.notna(row_names.iloc[1]) else 0,
                    'Free_Push_Reach': int(row_names.iloc[3]) if pd.notna(row_names.iloc[3]) else 0,
                    'Paid_WA_Reach': int(row_names.iloc[7]) if pd.notna(row_names.iloc[7]) else 0,
                    'Paid_SMS_Reach': int(row_names.iloc[4]) if pd.notna(row_names.iloc[4]) else 0,
                    'Paid_Email_Reach': int(row_names.iloc[5]) if pd.notna(row_names.iloc[5]) else 0
                })
            return pd.DataFrame(clean_data)
        except Exception as e:
            st.error(f"Error loading {sheet_name}: {e}")
            return None

    # 2. SELECTION UI
    st.divider()
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        view_type = st.radio(
            "Analyze Audience By:", 
            ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
        )
    
    df_active = load_and_stitch_volume(view_type)
    
    if df_active is not None:
        with col_b:
            target_selection = st.selectbox(f"Select Specific {view_type}", df_active['Name'].unique())
        
        # 3. ANALYSIS LOGIC
        data = df_active[df_active['Name'] == target_selection].iloc[0]
        
        st.divider()
        st.subheader(f"📈 Reach Insights for {target_selection}")
        
        # Key Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Base", f"{data['Total_Audience']:,}")
        m2.metric("FREE Reach (Mobile Push)", f"{data['Free_Push_Reach']:,}", help="Cost: ₹0.00")
        
        # Logic Y: Calculate the "Paid Gap"
        paid_gap = data['Paid_WA_Reach'] - data['Free_Push_Reach']
        m3.metric("Paid Incremental Lift", f"{paid_gap:,}", help="Users reachable ONLY by spending on WhatsApp/SMS")

        # 4. CHANNEL OPTIMIZATION TABLE
        st.write("### 🛠️ Cost vs. Reach Priority")
        optimizer_data = [
            {"Channel": "Mobile Push", "Reach": data['Free_Push_Reach'], "Cost": "Free (₹0)", "Priority": "1 (Primary)"},
            {"Channel": "WhatsApp", "Reach": data['Paid_WA_Reach'], "Cost": "Paid (High)", "Priority": "2 (Max Reach)"},
            {"Channel": "SMS", "Reach": data['Paid_SMS_Reach'], "Cost": "Paid (Mid)", "Priority": "3 (Fallback)"},
            {"Channel": "Email", "Reach": data['Paid_Email_Reach'], "Cost": "Paid (Low)", "Priority": "4 (Engagement)"}
        ]
        st.table(pd.DataFrame(optimizer_data))

        # 5. BUSINESS IMPACT PREDICTOR
        st.divider()
        st.subheader("🔮 Conversion & Revenue Forecast")
        
        p1, p2 = st.columns(2)
        with p1:
            conv_rate = st.slider("Expected Conversion Rate (%)", 0.05, 5.0, 1.0, 0.05)
        with p2:
            aov = st.number_input("Average Order Value (₹)", value=850)

        # Revenue Logic
        free_revenue = (data['Free_Push_Reach'] * (conv_rate / 100)) * aov
        total_potential_revenue = (data['Paid_WA_Reach'] * (conv_rate / 100)) * aov
        lift_revenue = total_potential_revenue - free_revenue

        st.success(f"**Total Revenue Potential:** ₹{int(total_potential_revenue):,}")
        st.info(f"**Incremental Revenue from Paid Channels:** ₹{int(lift_revenue):,}")
        st.caption("This forecast shows how much extra money is on the table by using WhatsApp to bridge the Push gap.")
