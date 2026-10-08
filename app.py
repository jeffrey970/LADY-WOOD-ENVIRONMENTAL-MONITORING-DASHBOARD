import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import folium
import plotly.express as px

from streamlit_folium import st_folium
from shapely.geometry import shape


# ============================================================
# LADYWOOD ENVIRONMENTAL RISK DASHBOARD
# Birmingham, United Kingdom
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

EXCEL_FILE = "Ladywood_4_Zone_Air_Risk_Final.xlsx"

BIRMINGHAM_GIS = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = f"{BIRMINGHAM_GIS}/14"
CRVA_LAYER = f"{BIRMINGHAM_GIS}/12"

PM25_LAYER = f"{BIRMINGHAM_GIS}/9"
NO2_LAYER = f"{BIRMINGHAM_GIS}/8"

FLUVIAL_LAYER = f"{BIRMINGHAM_GIS}/1547"
PLUVIAL_LAYER = f"{BIRMINGHAM_GIS}/1546"

GREENSPACE_LAYER = f"{BIRMINGHAM_GIS}/1549"
WOODLAND_LAYER = f"{BIRMINGHAM_GIS}/1552"

WHO_NO2 = 10.0
WHO_PM25 = 5.0


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
        color: #17324d;
    }

    .subtitle {
        font-size: 17px;
        color: #66727c;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        color: #17324d;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    .risk-high {
        background-color: #f8d7da;
        padding: 12px;
        border-radius: 8px;
        border-left: 6px solid #c62828;
    }

    .risk-medium {
        background-color: #fff3cd;
        padding: 12px;
        border-radius: 8px;
        border-left: 6px solid #e0a800;
    }

    .risk-low {
        background-color: #dff2e1;
        padding: 12px;
        border-radius: 8px;
        border-left: 6px solid #2e7d32;
    }

    .method-box {
        background-color: #f5f7f9;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #dce1e5;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCTIONS
# ============================================================

@st.cache_data(ttl=3600)
def get_arcgis_layer(url, where="1=1"):

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }

    response = requests.get(
        f"{url}/query",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:
        return gpd.GeoDataFrame()

    records = []

    for feature in data["features"]:

        properties = feature.get("properties", {})
        geometry = feature.get("geometry")

        if geometry:
            properties["geometry"] = shape(geometry)
            records.append(properties)

    if not records:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326"
    )


@st.cache_data(ttl=3600)
def get_brownfield_data():

    url = (
        "https://www.planning.data.gov.uk/"
        "api/1.0/entity.json"
    )

    params = {
        "dataset": "brownfield-land",
        "organisation-entity": "44",
        "limit": 1000
    }

    try:

        r = requests.get(
            url,
            params=params,
            timeout=60
        )

        if r.status_code != 200:
            return pd.DataFrame()

        data = r.json()

        if isinstance(data, dict):
            records = data.get("entities", data.get("results", []))
        else:
            records = data

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        return df

    except Exception:

        return pd.DataFrame()


@st.cache_data
def load_air_data():

    monthly = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Original_Monthly_Data"
    )

    zone = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Zone_Year_Calcs"
    )

    summary = pd.read_excel(
        EXCEL_FILE,
        sheet_name="Zone_Risk_Summary"
    )

    return monthly, zone, summary


def classify_air(no2, pm25):

    if pd.isna(no2) or pd.isna(pm25):
        return "UNKNOWN"

    no2_percent = no2 / WHO_NO2 * 100
    pm_percent = pm25 / WHO_PM25 * 100

    if no2_percent <= 100 and pm_percent <= 100:
        return "LOW"

    if no2_percent <= 150 and pm_percent <= 150:
        return "MODERATE"

    return "HIGH"


def risk_score(no2, pm25):

    no2_percent = no2 / WHO_NO2 * 100
    pm_percent = pm25 / WHO_PM25 * 100

    return (no2_percent + pm_percent) / 2


def risk_colour(risk):

    if risk == "HIGH":
        return "#c62828"

    if risk == "MODERATE":
        return "#f0b429"

    return "#2e7d32"


# ============================================================
# LOAD DATA
# ============================================================

try:

    monthly, zone, summary = load_air_data()

except Exception as error:

    st.error(
        "The Excel file could not be loaded. "
        "Make sure Ladywood_4_Zone_Air_Risk_Final.xlsx "
        "is in the same folder as app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">LADYWOOD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Environmental Risk & Development Dashboard '
    '— Birmingham, United Kingdom'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Dashboard controls")

years = sorted(
    monthly["Year"].dropna().unique().astype(int)
)

selected_year = st.sidebar.selectbox(
    "Year",
    years,
    index=len(years) - 1
)

zones = [
    "All zones",
    "City Centre",
    "Jewellery Quarter & Broad Street",
    "Park Central",
    "Remainder of Ladywood"
]

selected_zone = st.sidebar.selectbox(
    "Development area",
    zones
)

show_flooding = st.sidebar.checkbox(
    "Show flood-risk layer",
    True
)

show_air = st.sidebar.checkbox(
    "Show air-risk points",
    True
)

show_brownfield = st.sidebar.checkbox(
    "Show brownfield sites",
    True
)


# ============================================================
# REAL LADYWOOD BOUNDARY
# ============================================================

with st.spinner("Loading the official Ladywood boundary..."):

    ladywood = get_arcgis_layer(
        WARD_LAYER,
        "WARDNME LIKE '%Ladywood%'"
    )


if ladywood.empty:

    st.error(
        "The Birmingham GIS service did not return the Ladywood boundary."
    )

    st.stop()


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section-title">Ladywood environmental map</div>',
    unsafe_allow_html=True
)

map_center = [
    ladywood.geometry.centroid.y.mean(),
    ladywood.geometry.centroid.x.mean()
]

m = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="CartoDB positron"
)


# Actual Ladywood boundary
folium.GeoJson(
    ladywood.to_json(),
    name="Official Ladywood Ward",
    style_function=lambda feature: {
        "fillColor": "#dfeaf2",
        "color": "#17324d",
        "weight": 4,
        "fillOpacity": 0.18
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["WARDNME"],
        aliases=["Ward:"]
    )
).add_to(m)


# ============================================================
# CRVA DATA
# ============================================================

try:

    crva = get_arcgis_layer(
        CRVA_LAYER,
        "WARDNME LIKE '%Ladywood%'"
    )

except Exception:

    crva = gpd.GeoDataFrame()


if not crva.empty:

    for _, row in crva.iterrows():

        mean_score = row.get("MEAN")

        if pd.isna(mean_score):
            continue

        if mean_score >= 6.9:
            colour = "#c62828"

        elif mean_score >= 5.0:
            colour = "#f0b429"

        else:
            colour = "#2e7d32"

        folium.GeoJson(
            gpd.GeoSeries(
                [row.geometry],
                crs="EPSG:4326"
            ).to_json(),
            style_function=lambda feature,
            colour=colour: {
                "fillColor": colour,
                "color": "#555555",
                "weight": 1,
                "fillOpacity": 0.15
            }
        ).add_to(m)


# ============================================================
# AIR-RISK POINTS
# ============================================================

zone_2026 = zone[
    zone["Year"] == selected_year
].copy()


# These are representative screening locations,
# NOT claimed official sub-zone boundaries.
zone_locations = {

    "City Centre":
        (52.4805, -1.9018),

    "Jewellery Quarter & Broad Street":
        (52.4815, -1.9130),

    "Park Central":
        (52.4732, -1.9190),

    "Remainder of Ladywood":
        (52.4770, -1.9220)
}


if show_air:

    for _, row in zone_2026.iterrows():

        zone_name = row["Zone"]

        if zone_name not in zone_locations:
            continue

        if selected_zone != "All zones":
            if zone_name != selected_zone:
                continue

        lat, lon = zone_locations[zone_name]

        risk = row["Air_Risk_Class"]

        colour = risk_colour(risk)

        popup = f"""
        <b>{zone_name}</b><br>
        Air risk: <b>{risk}</b><br>
        NO₂: {row['NO2_Calculated']:.2f} µg/m³<br>
        PM2.5: {row['PM2_5_Calculated']:.2f} µg/m³<br>
        Risk score: {row['Air_Risk_Score']:.2f}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color="white",
            weight=2,
            fill=True,
            fill_color=colour,
            fill_opacity=0.95,
            popup=folium.Popup(
                popup,
                max_width=300
            )
        ).add_to(m)


# ============================================================
# BROWNFIELDS
# ============================================================

if show_brownfield:

    brownfields = [

        {
            "name":
                "IPL Site, Ladywood",
            "lat":
                52.480986,
            "lon":
                -1.931924,
            "ha":
                0.55
        },

        {
            "name":
                "IPL Site, Ladywood",
            "lat":
                52.481159,
            "lon":
                -1.930243,
            "ha":
                1.25
        },

        {
            "name":
                "Ledsam Street, Ladywood",
            "lat":
                52.480270,
            "lon":
                -1.924084,
            "ha":
                3.93
        },

        {
            "name":
                "Chamberlain Buildings, Corporation Street",
            "lat":
                52.484217,
            "lon":
                -1.892989,
            "ha":
                0.19
        },

        {
            "name":
                "Brindley Drive Multi-Storey Car Park",
            "lat":
                52.480642,
            "lon":
                -1.907971,
            "ha":
                0.33
        },

        {
            "name":
                "IPL Site, Ladywood",
            "lat":
                52.479030,
            "lon":
                -1.932896,
            "ha":
                3.62
        }
    ]

    for site in brownfields:

        if selected_zone != "All zones":
            # Brownfield points are shown as environmental evidence;
            # they are not forced into a zone where the boundary is unavailable.
            pass

        popup = f"""
        <b>Brownfield site</b><br>
        {site['name']}<br>
        Area: {site['ha']} ha
        """

        folium.CircleMarker(
            location=[
                site["lat"],
                site["lon"]
            ],
            radius=6,
            color="#1565c0",
            fill=True,
            fill_color="#1565c0",
            fill_opacity=0.85,
            popup=folium.Popup(
                popup,
                max_width=280
            )
        ).add_to(m)


# ============================================================
# FLOOD RISK
# ============================================================

if show_flooding:

    try:

        flood = get_arcgis_layer(
            PLUVIAL_LAYER,
            "1=1"
        )

        if not flood.empty:

            flood = gpd.clip(
                flood,
                ladywood
            )

            folium.GeoJson(
                flood.to_json(),
                name="Surface-water flood risk",
                style_function=lambda feature: {
                    "fillColor": "#4f81bd",
                    "color": "#2c5d91",
                    "weight": 1,
                    "fillOpacity": 0.30
                },
                tooltip="Surface-water flood-risk area"
            ).add_to(m)

    except Exception:
        pass


# ============================================================
# LEGEND
# ============================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 12px;
border: 1px solid #999;
border-radius: 6px;
font-size: 13px;
">

<b>Environmental indicators</b><br><br>

<span style="color:#c62828">●</span>
High air risk<br>

<span style="color:#f0b429">●</span>
Moderate air risk<br>

<span style="color:#2e7d32">●</span>
Lower air risk<br>

<span style="color:#1565c0">●</span>
Brownfield site<br>

<span style="color:#4f81bd">■</span>
Surface-water flood risk

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)

folium.LayerControl().add_to(m)


st_folium(
    m,
    width=None,
    height=600,
    returned_objects=[]
)


# ============================================================
# AIR QUALITY SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Air-quality evidence</div>',
    unsafe_allow_html=True
)


year_data = monthly[
    monthly["Year"] <= selected_year
].copy()


latest = monthly[
    monthly["Year"] == selected_year
].copy()


# ------------------------------------------------------------
# KPI VALUES
# ------------------------------------------------------------

annual_zone = zone[
    zone["Year"] == selected_year
].copy()


if selected_zone != "All zones":

    annual_zone = annual_zone[
        annual_zone["Zone"] == selected_zone
    ]


if not annual_zone.empty:

    highest = annual_zone.loc[
        annual_zone["Air_Risk_Score"].idxmax()
    ]

else:

    highest = zone[
        zone["Year"] == selected_year
    ].loc[
        zone[
            zone["Year"] == selected_year
        ]["Air_Risk_Score"].idxmax()
    ]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Highest priority zone",
        highest["Zone"]
    )


with col2:

    st.metric(
        "Risk classification",
        highest["Air_Risk_Class"]
    )


with col3:

    st.metric(
        "NO₂",
        f"{highest['NO2_Calculated']:.2f} µg/m³"
    )


with col4:

    st.metric(
        "PM2.5",
        f"{highest['PM2_5_Calculated']:.2f} µg/m³"
    )


# ============================================================
# TREND DATA
# ============================================================

st.markdown(
    "### Monthly pollution trend"
)


trend = monthly[
    monthly["Year"] <= selected_year
].copy()


trend["Date"] = pd.to_datetime(
    trend["Year_Month"].astype(str)
)


fig_no2 = px.line(
    trend,
    x="Date",
    y="NO2_Average",
    markers=True,
    title="Ladywood background-station NO₂"
)

fig_no2.add_hline(
    y=WHO_NO2,
    line_dash="dash",
    annotation_text="WHO annual guideline: 10 µg/m³"
)

fig_no2.update_layout(
    xaxis_title="Month",
    yaxis_title="NO₂ (µg/m³)"
)

st.plotly_chart(
    fig_no2,
    use_container_width=True
)


fig_pm = px.line(
    trend,
    x="Date",
    y="PM2_5_Average",
    markers=True,
    title="Ladywood background-station PM2.5"
)

fig_pm.add_hline(
    y=WHO_PM25,
    line_dash="dash",
    annotation_text="WHO annual guideline: 5 µg/m³"
)

fig_pm.update_layout(
    xaxis_title="Month",
    yaxis_title="PM2.5 (µg/m³)"
)

st.plotly_chart(
    fig_pm,
    use_container_width=True
)


# ============================================================
# ZONE COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">Development-zone comparison</div>',
    unsafe_allow_html=True
)


comparison = zone[
    zone["Year"] == selected_year
].copy()


comparison_display = comparison[
    [
        "Zone",
        "Air_Multiplier",
        "NO2_Calculated",
        "PM2_5_Calculated",
        "NO2_WHO_Percent",
        "PM2_5_WHO_Percent",
        "Air_Risk_Score",
        "Air_Risk_Class"
    ]
].copy()


comparison_display.columns = [
    "Zone",
    "Screening multiplier",
    "NO₂",
    "PM2.5",
    "NO₂ / WHO",
    "PM2.5 / WHO",
    "Risk score",
    "Risk"
]


st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RISK SCORE CHART
# ============================================================

fig_risk = px.bar(
    comparison,
    x="Zone",
    y="Air_Risk_Score",
    color="Air_Risk_Class",
    title=f"Air-quality screening score by development area — {selected_year}"
)

fig_risk.update_layout(
    xaxis_title="Development area",
    yaxis_title="Screening risk score"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# FLOODING AND CLIMATE
# ============================================================

st.markdown(
    '<div class="section-title">Flooding and climate evidence</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)


with c1:

    st.markdown(
        """
        <div class="method-box">

        <h3>Flooding</h3>

        Birmingham's Strategic Flood Risk Assessment identifies
        Ladywood as an area susceptible to surface-water flooding.

        The dashboard therefore treats surface-water flood exposure
        as a separate environmental risk layer rather than inventing
        flood measurements.

        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        """
        <div class="method-box">

        <h3>Climate vulnerability</h3>

        Birmingham's Climate Risk and Vulnerability Assessment
        provides spatial information covering climate vulnerability,
        air pollution, flooding, greenspace, woodland and temperature.

        The dashboard uses these layers as spatial environmental
        evidence.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BROWNFIELD SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Brownfield and land-condition evidence</div>',
    unsafe_allow_html=True
)

brownfield_table = pd.DataFrame(
    [
        [
            "IPL Site, Ladywood",
            "N717H",
            0.55
        ],
        [
            "IPL Site, Ladywood",
            "N717D",
            1.25
        ],
        [
            "IPL Site, Ladywood",
            "N718",
            3.62
        ],
        [
            "Ledsam Street, Ladywood",
            "CC1",
            3.93
        ],
        [
            "Chamberlain Buildings, Corporation Street",
            "CC447",
            0.19
        ],
        [
            "Brindley Drive Multi-Storey Car Park",
            "CC445",
            0.33
        ]
    ],
    columns=[
        "Site",
        "Reference",
        "Area (ha)"
    ]
)

st.dataframe(
    brownfield_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# OVERALL ENVIRONMENTAL PRIORITY
# ============================================================

st.markdown(
    '<div class="section-title">Environmental priority assessment</div>',
    unsafe_allow_html=True
)


st.write(
    "The final priority should not be presented as a measured "
    "pollution concentration. It is an engineering screening "
    "assessment combining available environmental evidence."
)


# Air score normalisation
priority = comparison.copy()

priority["Air_Index"] = (
    priority["Air_Risk_Score"] /
    priority["Air_Risk_Score"].max()
)


# CRVA context
crva_score = 7.14

priority["Climate_Index"] = crva_score / 10


# Because the CRVA score applies at ward level,
# it is deliberately not pretending to distinguish
# between the four areas.
priority["Overall_Screening_Index"] = (
    priority["Air_Index"] * 0.70
    +
    priority["Climate_Index"] * 0.30
)


priority = priority.sort_values(
    "Overall_Screening_Index",
    ascending=False
)


priority_display = priority[
    [
        "Zone",
        "Air_Risk_Score",
        "Air_Risk_Class",
        "Overall_Screening_Index"
    ]
].copy()


priority_display["Overall_Screening_Index"] = (
    priority_display[
        "Overall_Screening_Index"
    ].round(3)
)


st.dataframe(
    priority_display,
    use_container_width=True,
    hide_index=True
)


top_priority = priority.iloc[0]


if top_priority["Air_Risk_Class"] == "HIGH":

    st.markdown(
        f"""
        <div class="risk-high">

        <b>Highest screening priority:</b>
        {top_priority['Zone']}

        <br><br>

        This area has the highest air-quality screening score
        in the available zone analysis.

        </div>
        """,
        unsafe_allow_html=True
    )

elif top_priority["Air_Risk_Class"] == "MODERATE":

    st.markdown(
        f"""
        <div class="risk-medium">

        <b>Highest screening priority:</b>
        {top_priority['Zone']}

        <br><br>

        This area should receive further investigation because
        it has the highest available air-quality screening score.

        </div>
        """,
        unsafe_allow_html=True
    )

else:

    st.markdown(
        f"""
        <div class="risk-low">

        <b>Highest screening priority:</b>
        {top_priority['Zone']}

        <br><br>

        Current available evidence does not indicate a high
        air-quality screening category.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">Engineering interpretation</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="method-box">

    <h3>How the dashboard makes the decision</h3>

    <b>Step 1 — Measure</b><br>
    Use the available Ladywood background-station air-quality data.

    <br><br>

    <b>Step 2 — Locate</b><br>
    Use the real Birmingham GIS Ladywood boundary and environmental
    spatial layers.

    <br><br>

    <b>Step 3 — Identify environmental stressors</b><br>
    Examine air pollution, flooding, climate vulnerability,
    greenspace and brownfield evidence.

    <br><br>

    <b>Step 4 — Compare development areas</b><br>
    Apply the project's documented air screening calculation
    to the four development areas.

    <br><br>

    <b>Step 5 — Prioritise</b><br>
    The area with the strongest combined evidence becomes the
    first area recommended for detailed investigation and
    environmental intervention.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section-title">Important data limitations</div>',
    unsafe_allow_html=True
)

st.warning(
    """
    The dashboard distinguishes between measured data and screening
    estimates.

    The air-quality station represents background conditions for
    Ladywood. The four development-area air values are therefore
    screening estimates based on the project multipliers; they are
    not direct measurements at every location.

    The Birmingham CRVA provides real spatial environmental evidence,
    while the brownfield records identify documented sites.

    Flood-risk polygons show mapped risk/exposure; they should not
    be interpreted as a prediction of the exact depth of flooding
    at a particular building.
    """
)


# ============================================================
# SOURCES
# ============================================================

st.markdown(
    '<div class="section-title">Data sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **Birmingham City Council — Climate Risk and Vulnerability Assessment**

    https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer

    **Birmingham Strategic Flood Risk Assessment**

    https://www.birmingham.gov.uk/download/downloads/id/1203/level_1_strategic_flood_risk_assessment.pdf

    **UK Government Planning Data — Brownfield land**

    https://www.planning.data.gov.uk/

    **Project air-quality dataset**

    Ladywood_4_Zone_Air_Risk_Final.xlsx

    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Ladywood Environmental Risk Dashboard | "
    "Wits Mining Engineering project | "
    "Environmental screening tool"
)
