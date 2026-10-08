import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Ladywood Civic Digital Twin Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stAlert { border-radius: 8px; }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATA GENERATION ENGINE (Simulating Ladywood Sensor Stream)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=5)
def generate_sensor_data():
    now = datetime.now()
    timestamps = [now - timedelta(minutes=i*5) for i in range(60)][::-1]
    
    # Base values reflecting Ladywood environmental parameters
    np.random.seed(int(time.time()) % 1000)
    
    data = pd.DataFrame({
        "Timestamp": timestamps,
        # Urban Heat Island (UHI) Temperature (°C)
        "Temp_LinkRoad": np.random.normal(31.5, 0.8, 60),  # High-density block
        "Temp_Reservoir": np.random.normal(26.0, 0.5, 60), # Green baseline (Edgbaston Res)
        # Canal & Water Quality
        "Canal_Turbidity_NTU": np.random.normal(12.0, 3.5, 60),
        "Canal_pH": np.random.normal(7.2, 0.3, 60),
        "Water_Level_Meters": np.random.normal(1.4, 0.15, 60),
        # Soil & Community Growing Beds
        "Soil_Moisture_Pct": np.random.normal(22.0, 4.0, 60),
        # Energy Micro-Grid (kW)
        "Solar_PV_Output": np.clip(np.random.normal(45.0, 15.0, 60), 0, None)
    })
    
    return data

df = generate_sensor_data()
latest = df.iloc[-1]

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS & LOCALIZATION
# ------------------------------------------------------------------------------
st.sidebar.title("🌱 Ladywood Civic Twin")
st.sidebar.caption("Neighbourhood Doughnut Environmental Engine")

# Language Selection for Digital Inclusion
language = st.sidebar.selectbox(
    "Select Language / Dil Chuniye", 
    ["English", "Urdu (اردو)", "Punjabi (ਪੰਜਾਬੀ)", "Bengali (বাংলা)", "Polish (Polski)"]
)

st.sidebar.divider()

# Simulation Overrides (For Testing Automated Interventions)
st.sidebar.subheader("⚙️ Local Scenario Injector")
sim_heatwave = st.sidebar.checkbox("Simulate Extreme Heat Spike")
sim_spill = st.sidebar.checkbox("Simulate CSO Water Spill Event")

if sim_heatwave:
    latest["Temp_LinkRoad"] = 36.8
if sim_spill:
    latest["Canal_Turbidity_NTU"] = 48.5
    latest["Canal_pH"] = 5.8

st.sidebar.divider()
st.sidebar.info("Connected to LoRaWAN Gateway: Ladywood Public Square Node #01")

# ------------------------------------------------------------------------------
# HEADER SECTION
# ------------------------------------------------------------------------------
st.title("Ladywood Interactive Environmental Dashboard")
st.markdown("Real-time telemetry, automated micro-interventions, and ecological boundary tracking.")

# ------------------------------------------------------------------------------
# AUTOMATED DECISION ENGINE (AUTOMATED ALERTS)
# ------------------------------------------------------------------------------
uhi_diff = latest["Temp_LinkRoad"] - latest["Temp_Reservoir"]

if latest["Temp_LinkRoad"] > 35.0 or uhi_diff > 5.0:
    st.error(f"""
        🔥 **AUTOMATED ALERT: URBAN HEAT ISLAND CRITICAL**  
        * **Link Road High-Density Block:** {latest['Temp_LinkRoad']:.1f}°C (Differential: +{uhi_diff:.1f}°C vs Reservoir)  
        * **Action Triggered:** Public Misting Systems Activated at Public Square | SMS Heat Warnings Dispatched to Vulnerable Households.
    """)
elif latest["Canal_Turbidity_NTU"] > 35.0 or latest["Canal_pH"] < 6.0:
    st.warning(f"""
        ⚠️ **AUTOMATED ALERT: CANAL WATER CONTAMINATION DETECTED**  
        * **Turbidity:** {latest['Canal_Turbidity_NTU']:.1f} NTU | **pH:** {latest['Canal_pH']:.2f}  
        * **Action Triggered:** Automated Runoff Isolation Gate Closed at Canal Loop | Canoeists & Water Users Notified via SMS.
    """)
else:
    st.success("✅ **SYSTEM HEALTH NORMAL:** Environmental boundaries within safe operational limits.")

st.divider()

# ------------------------------------------------------------------------------
# KEY METRICS ROW
# ------------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Link Road Temperature", 
        value=f"{latest['Temp_LinkRoad']:.1f} °C", 
        delta=f"+{uhi_diff:.1f} °C vs Green Pocket",
        delta_color="inverse"
    )

with col2:
    st.metric(
        label="Canal Turbidity", 
        value=f"{latest['Canal_Turbidity_NTU']:.1f} NTU", 
        delta="-3.2 NTU (1 hr)" if not sim_spill else "+28.5 NTU CRISIS",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="Growing Soil Moisture", 
        value=f"{latest['Soil_Moisture_Pct']:.1f} %", 
        delta="Optimal (20-30%)" if 20 <= latest['Soil_Moisture_Pct'] <= 30 else "Low Irrigation Needed"
    )

with col4:
    st.metric(
        label="Community Solar Gen.", 
        value=f"{latest['Solar_PV_Output']:.1f} kW", 
        delta="+8.4 kW Grid Feed"
    )

# ------------------------------------------------------------------------------
# DASHBOARD TABS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🌡️ Microclimate & Heat", "🌊 Water & Canal Health", "📍 Interactive Spatial Map"])

# TAB 1: URBAN HEAT ISLAND
with tab1:
    st.subheader("Urban Heat Island (UHI) Dynamics")
    st.caption("Comparing built-up high-density residential blocks against reference green spaces (Edgbaston Reservoir).")
    
    fig_heat = px.line(
        df, 
        x="Timestamp", 
        y=["Temp_LinkRoad", "Temp_Reservoir"],
        labels={"value": "Temperature (°C)", "variable": "Sensor Location"},
        color_discrete_map={"Temp_LinkRoad": "#e74c3c", "Temp_Reservoir": "#2ecc71"},
        title="60-Minute Temperature Comparison Stream"
    )
    fig_heat.update_layout(hovermode="x unified", legend_title_text="Sensor Location")
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 2: WATER & CANAL MONITORING
with tab2:
    st.subheader("Canal Loop & Runoff Water Quality")
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        fig_water = px.area(
            df, 
            x="Timestamp", 
            y="Canal_Turbidity_NTU",
            title="Canal Turbidity Levels (NTU)",
            color_discrete_sequence=["#3498db"]
        )
        fig_water.add_hline(y=35.0, line_dash="dash", line_color="red", annotation_text="Pollution Trigger Limit")
        st.plotly_chart(fig_water, use_container_width=True)
        
    with col_w2:
        fig_ph = px.line(
            df, 
            x="Timestamp", 
            y="Canal_pH",
            title="Real-Time pH Levels",
            color_discrete_sequence=["#9b59b6"]
        )
        fig_ph.add_hline(y=6.5, line_dash="dash", line_color="orange", annotation_text="Acidity Alert Threshold")
        st.plotly_chart(fig_ph, use_container_width=True)

# TAB 3: SPATIAL MAP
with tab3:
    st.subheader("Ladywood Sensor Deployment Map")
    st.caption("Live spatial network nodes across Ladywood ward.")
    
    # Coordinates centered around Ladywood, Birmingham
    map_data = pd.DataFrame({
        'lat': [52.4790, 52.4765, 52.4821, 52.4740],
        'lon': [-1.9210, -1.9300, -1.9150, -1.9100],
        'Node_Name': [
            'Link Road High-Density Block', 
            'Edgbaston Reservoir Baseline', 
            'Ladywood Canal Junction', 
            'Neighbourhood Public Square Hub'
        ],
        'Status': ['Active (High Temp)', 'Active (Normal)', 'Active (Monitoring)', 'Active Hub']
    })
    
    fig_map = px.scatter_mapbox(
        map_data,
        lat="lat",
        lon="lon",
        hover_name="Node_Name",
        hover_data=["Status"],
        zoom=13,
        height=450
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------------------------------------------------------
# FOOTER & PHYSICAL E-INK DISPLAY SIMULATOR
# ------------------------------------------------------------------------------
st.divider()
st.subheader("📟 Public Hub E-Ink Physical Display Preview")
st.caption("Rendering low-power output for non-digital screen terminals at public hubs.")

eink_container = st.container()
with eink_container:
    st.code(f"""
============================================================
           LADYWOOD COMMUNITY ENVIRONMENTAL MONITOR         
============================================================
TIME: {latest['Timestamp'].strftime('%H:%M:%S')} | STATUS: {'⚠️ ALERT ACTIVE' if (sim_heatwave or sim_spill) else '✅ SAFE'}

[HEAT INDEX]   Link Road: {latest['Temp_LinkRoad']:.1f}°C  | Reservoir: {latest['Temp_Reservoir']:.1f}°C
[WATER QUAL]   Turbidity: {latest['Canal_Turbidity_NTU']:.1f} NTU | pH: {latest['Canal_pH']:.2f}
[SOLAR GEN ]   Current Output: {latest['Solar_PV_Output']:.1f} kW

NOTIFICATION: {'Extreme Heat Detected. Cool Down Hub Open at Public Square.' if sim_heatwave else 'Air and Water levels within safe healthy bounds today.'}
============================================================
    """, language="text")
