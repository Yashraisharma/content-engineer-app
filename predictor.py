import streamlit as st
import pandas as pd

def run_page():
    st.header("📊 Segment Analysis & Reach Predictor")
    st.markdown("#### Logic Y: Volume-First Strategic Planning")
    
    # 1. THE DATA SOURCE
    # Note: Using the raw link to handle the double extension in your filename
    EXCEL_URL = "https://github.com/Yashraisharma/content-engineer-app/raw/main/cohort_sheets.xlsx.xlsx"

    @st.cache_data
    def load_and_stitch_volume(sheet_name):
        try:
            # We use openpyxl as the engine for Excel
            df = pd.read_excel(EXCEL_URL, sheet_name=sheet_name, engine='openpyxl').dropna(how='all').reset_index(drop=True)
            clean_data = []
            
            # This loop handles the 2-row "Stacked" format from your exports
            for i in range(0, len(df), 2):
                if i+1 >= len(df): break
                row_names = df.iloc[i]
                
                # Filter out system header rows if they repeat
                first_val = str(row_names.iloc[0]).lower()
                if first_val in ['city', 'category', 'segment', 'status', 'sku_id']:
                    continue
                
                # Logic Y: Capturing Raw Headcounts
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
            st.error(f"⚠️ Error accessing sheet '{sheet_name}': {e}")
            return None

    # 2. STRATEGIC SELECTION UI
    st.divider()
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        view_type = st.radio(
            "Targeting Dimension:", 
            ["top 6 cities", "pharma_focus _category_new", "Daily_pharma_portfolio_segment"]
        )
    
    # Load data for the selected stream
    df_active = load_and_stitch_volume(view_type)
    
    if df_active is not None and not df_active.empty:
        with col_b:
            target_selection = st.selectbox(f"Select {view_type.split('_')[0].title()}", df_active['Name'].unique())
        
        # 3. VOLUME & REACH ANALYSIS
        data = df_active[df_active['Name'] == target_selection].iloc[0]
        
        st.divider()
        st.subheader(f"📈 Reach Insights: {target_selection}")
        
        # Key Metrics Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Addressable Base", f"{data['Total_Audience']:,}")
        m2.metric("FREE Reach (Push)", f"{data['Free_Push_Reach']:,}", "₹0.00 Cost")
        
        # Calculate the "Paid Gap" (WhatsApp Lift)
        paid_lift = data['Paid_WA_Reach'] - data['Free_Push_Reach']
        m3.metric("Paid Incremental Lift", f"{paid_lift:,}", "via WhatsApp/SMS")

        # 4. CHANNEL PRIORITY MATRIX
        st.write("### 🛠️ Channel Optimization Strategy")
        optimizer_data = [
            {"Priority": "1 (Primary)", "Channel": "Mobile Push", "Users": f"{data['Free_Push_Reach']:,}", "Unit Cost": "₹0.00"},
            {"Priority": "2 (Secondary)", "Channel": "WhatsApp", "Users": f"{data['Paid_WA_Reach']:,}", "Unit Cost": "Paid (High)"},
            {"Priority": "3 (Fallback)", "Channel": "SMS", "Users": f"{data['Paid_SMS_Reach']:,}", "Unit Cost": "Paid (Mid)"},
            {"Priority": "4 (Awareness)", "Channel": "Email", "Users": f"{data['Paid_Email_Reach']:,}", "Unit Cost": "Paid (Low)"}
        ]
        st.table(pd.DataFrame(optimizer_data))

        # 5. CONVERSION & REVENUE PREDICTOR
        st.divider()
        st.subheader("🔮 Growth Forecast (AOV & Conversion)")
        
        p1, p2 = st.columns(2)
        with p1:
            conv_rate = st.slider("Conversion Rate (%)", 0.1, 5.0, 1.0, 0.1)
        with p2:
            aov = st.number_input("Average Order Value (₹)", value=800)

        # Revenue Forecast Logic
        base_revenue = (data['Free_Push_Reach'] * (conv_rate / 100)) * aov
        total_revenue = (data['Paid_WA_Reach'] * (conv_rate / 100)) * aov
        incremental_revenue = total_revenue - base_revenue

        st.success(f"**Total Revenue Potential:** ₹{int(total_revenue):,}")
        st.info(f"**Incremental Value from Paid Lift:** ₹{int(incremental_revenue):,}")
        st.caption(f"This logic calculates that using WhatsApp adds **{paid_lift:,}** reachable users, creating an extra **₹{int(incremental_revenue):,}** in revenue potential.")

    else:
        st.warning("Please ensure your Excel file is uploaded and the URL is correct.")
