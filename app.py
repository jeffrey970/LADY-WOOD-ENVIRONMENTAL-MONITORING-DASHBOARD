import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import folium
import plotly.express as px

from streamlit_folium import st_folium
from shapely.geometry import shape
from shapely.ops import unary_union


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# OFFICIAL BIRMINGHAM DATA SERVICES
# ============================================================

CRVA_SERVICE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

BOUNDARY_LAYER = f"{CRVA_SERVICE}/14"

LSOA_CRVA_LAYER = f"{CRVA_SERVICE}/15"

FLOOD_ZONE_3_LAYER = f"{CRVA_SERVICE}/109"

SURFACE_FLOOD_LAYER = f"{CRVA_SERVICE}/1546"

GREENSPACE_LAYER = f"{CRVA_SERVICE}/1549"

WOODLAND_LAYER = f"{CRVA_SERVICE}/1552"

BROWNFIELD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)


# ============================================================
# GENERAL SETTINGS
# ============================================================

REQUEST_TIMEOUT = 60


# ============================================================
# ARC GIS DATA FUNCTION
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def arcgis_query(
    layer_url,
    where="1=1",
    out_fields="*",
    return_geometry=True,
    out_sr=4326
):

    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": str(return_geometry).lower(),
        "outSR": out_sr,
        "f": "geojson"
    }

    response = requests.get(
        layer_url + "/query",
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(str(data["error"]))

    features = data.get("features", [])

    if not features:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    rows = []

    for feature in features:

        properties = feature.get("properties", {})
        geometry = feature.get("geometry")

        row = properties.copy()

        if geometry:
            row["geometry"] = shape(geometry)
        else:
            row["geometry"] = None

        rows.append(row)

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return gdf


# ============================================================
# LOAD LADYWOOD BOUNDARY
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_ladywood():

    wards = arcgis_query(
        BOUNDARY_LAYER,
        where="1=1",
        out_fields="WARDNME,WARD_CODE"
    )

    if wards.empty:
        raise RuntimeError(
            "Birmingham's official ward boundary service returned no data."
        )

    wards["WARDNME"] = wards["WARDNME"].astype(str).str.strip()

    ladywood = wards[
        wards["WARDNME"].str.lower() == "ladywood"
    ].copy()

    if ladywood.empty:

        ladywood = wards[
            wards["WARDNME"].str.contains(
                "ladywood",
                case=False,
                na=False
            )
        ].copy()

    if ladywood.empty:

        names = sorted(
            wards["WARDNME"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Ladywood could not be identified in the official "
            f"Birmingham ward dataset.\n\nAvailable wards:\n{names}"
        )

    return ladywood


# ============================================================
# LOAD LSOA CRVA DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_lsoa_crva():

    data = arcgis_query(
        LSOA_CRVA_LAYER,
        where="1=1",
        out_fields=(
            "LSOA21CD,LSOA21NM,MIN,MAX,MEAN,STD,"
            "MEDIAN,MINIMUM_RISK,AVERAGE_RISK,MAXIMUM_RISK"
        )
    )

    if data.empty:
        raise RuntimeError(
            "The official Birmingham LSOA CRVA layer returned no data."
        )

    return data


# ============================================================
# LOAD FLOOD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_flood_zone_3():

    return arcgis_query(
        FLOOD_ZONE_3_LAYER,
        where="1=1",
        out_fields="origin,flood_zone,flood_sour"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def load_surface_flood():

    return arcgis_query(
        SURFACE_FLOOD_LAYER,
        where="1=1",
        out_fields="pub_date,tile_id"
    )


# ============================================================
# LOAD BROWNFIELD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_brownfield():

    return arcgis_query(
        BROWNFIELD_LAYER,
        where="1=1",
        out_fields=(
            "SiteReference,SiteNameAddress,"
            "Hectares,OwnershipStatus,PlanningStatus,"
            "PermissionType,LastUpdatedDate"
        )
    )


# ============================================================
# SPATIAL CALCULATION
# ============================================================

def calculate_ladywood_indicators(
    ladywood,
    lsoa_data,
    flood3,
    surface_flood,
    brownfield
):

    ladywood_polygon = unary_union(ladywood.geometry)

    # --------------------------------------------------------
    # Keep only LSOAs that intersect Ladywood
    # --------------------------------------------------------

    lsoa = lsoa_data[
        lsoa_data.geometry.intersects(ladywood_polygon)
    ].copy()

    if lsoa.empty:
        raise RuntimeError(
            "No official LSOA areas were found inside Ladywood."
        )

    # --------------------------------------------------------
    # Project everything into British National Grid.
    # This makes area calculations metres / m².
    # --------------------------------------------------------

    lsoa = lsoa.to_crs(27700)

    ladywood_projected = ladywood.to_crs(27700)

    ladywood_polygon_projected = unary_union(
        ladywood_projected.geometry
    )

    # Keep only the actual portion inside Ladywood
    lsoa["geometry"] = lsoa.geometry.intersection(
        ladywood_polygon_projected
    )

    lsoa = lsoa[
        ~lsoa.geometry.is_empty
    ].copy()

    lsoa["LSOA_Area_m2"] = lsoa.geometry.area

    lsoa["LSOA_Area_ha"] = (
        lsoa["LSOA_Area_m2"] / 10000
    )

    # --------------------------------------------------------
    # FLOOD ZONE 3
    # --------------------------------------------------------

    flood3 = flood3.to_crs(27700)

    if not flood3.empty:

        flood3_union = unary_union(
            flood3.geometry
        )

        lsoa["Flood_Zone_3_Area_m2"] = (
            lsoa.geometry
            .intersection(flood3_union)
            .area
        )

    else:

        lsoa["Flood_Zone_3_Area_m2"] = 0.0

    lsoa["Flood_Zone_3_Percent"] = (
        lsoa["Flood_Zone_3_Area_m2"]
        / lsoa["LSOA_Area_m2"]
        * 100
    )

    # --------------------------------------------------------
    # SURFACE FLOODING
    # --------------------------------------------------------

    surface_flood = surface_flood.to_crs(27700)

    if not surface_flood.empty:

        surface_union = unary_union(
            surface_flood.geometry
        )

        lsoa["Surface_Flood_Area_m2"] = (
            lsoa.geometry
            .intersection(surface_union)
            .area
        )

    else:

        lsoa["Surface_Flood_Area_m2"] = 0.0

    lsoa["Surface_Flood_Percent"] = (
        lsoa["Surface_Flood_Area_m2"]
        / lsoa["LSOA_Area_m2"]
        * 100
    )

    # --------------------------------------------------------
    # COMBINED FLOOD EXPOSURE
    #
    # We use the larger mapped flood coverage rather than
    # adding the two layers together, preventing double
    # counting of overlapping areas.
    # --------------------------------------------------------

    lsoa["Flood_Exposure_Percent"] = lsoa[
        [
            "Flood_Zone_3_Percent",
            "Surface_Flood_Percent"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # BROWNFIELD
    # --------------------------------------------------------

    brownfield = brownfield.to_crs(27700)

    if not brownfield.empty:

        brownfield_union = unary_union(
            brownfield.geometry
        )

        lsoa["Brownfield_Area_m2"] = (
            lsoa.geometry
            .intersection(brownfield_union)
            .area
        )

        # Count individual brownfield sites touching each LSOA
        site_counts = []

        for geom in lsoa.geometry:

            count = int(
                brownfield.geometry.intersects(geom).sum()
            )

            site_counts.append(count)

        lsoa["Brownfield_Site_Count"] = site_counts

    else:

        lsoa["Brownfield_Area_m2"] = 0.0
        lsoa["Brownfield_Site_Count"] = 0

    lsoa["Brownfield_Percent"] = (
        lsoa["Brownfield_Area_m2"]
        / lsoa["LSOA_Area_m2"]
        * 100
    )

    # --------------------------------------------------------
    # CRVA
    #
    # MEAN is an official Birmingham CRVA value.
    # We DO NOT invent another CRVA value.
    # --------------------------------------------------------

    lsoa["CRVA_Mean"] = pd.to_numeric(
        lsoa["MEAN"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # NORMALISED VALUES
    #
    # These are only used to put the different measured
    # indicators onto comparable 0-1 scales.
    # --------------------------------------------------------

    crva_min = lsoa["CRVA_Mean"].min()
    crva_max = lsoa["CRVA_Mean"].max()

    if crva_max > crva_min:

        lsoa["CRVA_Index"] = (
            (lsoa["CRVA_Mean"] - crva_min)
            / (crva_max - crva_min)
        )

    else:

        lsoa["CRVA_Index"] = 0.0

    # Flood exposure relative to the highest observed
    # Ladywood LSOA value.

    max_flood = lsoa["Flood_Exposure_Percent"].max()

    if max_flood > 0:

        lsoa["Flood_Index"] = (
            lsoa["Flood_Exposure_Percent"]
            / max_flood
        )

    else:

        lsoa["Flood_Index"] = 0.0

    # Brownfield exposure relative to highest observed
    # Ladywood LSOA value.

    max_brownfield = lsoa["Brownfield_Percent"].max()

    if max_brownfield > 0:

        lsoa["Brownfield_Index"] = (
            lsoa["Brownfield_Percent"]
            / max_brownfield
        )

    else:

        lsoa["Brownfield_Index"] = 0.0

    # --------------------------------------------------------
    # LADYWOOD ENGINEERING SCREENING SCORE
    #
    # This is NOT a Birmingham Council score.
    #
    # It is a transparent project screening calculation:
    #
    # 50% CRVA
    # 30% flood exposure
    # 20% brownfield exposure
    # --------------------------------------------------------

    lsoa["Screening_Score"] = (
        0.50 * lsoa["CRVA_Index"]
        + 0.30 * lsoa["Flood_Index"]
        + 0.20 * lsoa["Brownfield_Index"]
    )

    lsoa["Screening_Percent"] = (
        lsoa["Screening_Score"] * 100
    )

    # --------------------------------------------------------
    # COLOUR CLASSIFICATION
    # --------------------------------------------------------

    def classify(score):

        if score >= 0.67:
            return "HIGH"

        elif score >= 0.34:
            return "MODERATE"

        else:
            return "LOW"

    lsoa["Risk_Level"] = (
        lsoa["Screening_Score"]
        .apply(classify)
    )

    # --------------------------------------------------------
    # CENTROIDS FOR THE MAP DOTS
    # --------------------------------------------------------

    lsoa["centroid"] = lsoa.geometry.centroid

    lsoa["Latitude"] = (
        lsoa["centroid"].to_crs(4326).y
    )

    lsoa["Longitude"] = (
        lsoa["centroid"].to_crs(4326).x
    )

    return lsoa


# ============================================================
# START DASHBOARD
# ============================================================

st.title("LADYWOOD")

st.subheader(
    "Environmental Risk & Climate Dashboard"
)

st.caption(
    "Ladywood, Birmingham, United Kingdom"
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner("Loading official Birmingham environmental data..."):

        ladywood = load_ladywood()

        lsoa_data = load_lsoa_crva()

        flood3 = load_flood_zone_3()

        surface_flood = load_surface_flood()

        brownfield = load_brownfield()

        indicators = calculate_ladywood_indicators(
            ladywood,
            lsoa_data,
            flood3,
            surface_flood,
            brownfield
        )

except Exception as error:

    st.error(
        "The dashboard could not load the Birmingham data."
    )

    st.exception(error)

    st.stop()


# ============================================================
# LADYWOOD SUMMARY
# ============================================================

ladywood_area_ha = (
    ladywood.to_crs(27700)
    .geometry
    .area
    .sum()
    / 10000
)

flood3_area_ha = (
    flood3.to_crs(27700)
    .overlay(
        ladywood.to_crs(27700),
        how="intersection"
    )
    .geometry
    .area
    .sum()
    / 10000
    if not flood3.empty
    else 0
)

brownfield_in_ladywood = (
    brownfield.to_crs(27700)
    .overlay(
        ladywood.to_crs(27700),
        how="intersection"
    )
    if not brownfield.empty
    else gpd.GeoDataFrame()
)

brownfield_area_ha = (
    brownfield_in_ladywood.geometry.area.sum()
    / 10000
    if not brownfield_in_ladywood.empty
    else 0
)


# ============================================================
# INDICATOR CARDS
# ============================================================

st.header("Ladywood Environmental Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Ladywood area",
    f"{ladywood_area_ha:,.1f} ha"
)

c2.metric(
    "Flood Zone 3",
    f"{flood3_area_ha:,.1f} ha"
)

c3.metric(
    "Brownfield area",
    f"{brownfield_area_ha:,.1f} ha"
)

c4.metric(
    "Mapped areas assessed",
    f"{len(indicators)}"
)


# ============================================================
# MAP
# ============================================================

st.header("Ladywood Environmental Risk Map")

st.write(
    "The coloured points represent named 2021 LSOA areas "
    "inside Ladywood. They are not artificial development zones."
)

ladywood_wgs84 = ladywood.to_crs(4326)

centre = ladywood_wgs84.geometry.union_all().centroid

map_center = [
    centre.y,
    centre.x
]

m = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="OpenStreetMap",
    control_scale=True
)


# ------------------------------------------------------------
# LADYWOOD BOUNDARY
# ------------------------------------------------------------

folium.GeoJson(
    ladywood_wgs84.to_json(),
    name="Ladywood boundary",
    style_function=lambda feature: {
        "fillColor": "#ffffff",
        "color": "#000000",
        "weight": 4,
        "fillOpacity": 0.05
    },
    tooltip="Ladywood ward boundary"
).add_to(m)


# ------------------------------------------------------------
# FLOOD ZONE 3
# ------------------------------------------------------------

if not flood3.empty:

    flood3_map = flood3.to_crs(4326)

    folium.GeoJson(
        flood3_map.to_json(),
        name="Flood Zone 3",
        style_function=lambda feature: {
            "fillColor": "#0066ff",
            "color": "#0044aa",
            "weight": 1,
            "fillOpacity": 0.30
        },
        tooltip="Birmingham Flood Zone 3"
    ).add_to(m)


# ------------------------------------------------------------
# SURFACE FLOODING
# ------------------------------------------------------------

if not surface_flood.empty:

    surface_map = surface_flood.to_crs(4326)

    folium.GeoJson(
        surface_map.to_json(),
        name="Surface flooding",
        style_function=lambda feature: {
            "fillColor": "#00a6d6",
            "color": "#007c99",
            "weight": 1,
            "fillOpacity": 0.18
        },
        tooltip="Birmingham mapped surface flooding"
    ).add_to(m)


# ------------------------------------------------------------
# BROWNFIELD
# ------------------------------------------------------------

if not brownfield.empty:

    brownfield_map = brownfield.to_crs(4326)

    folium.GeoJson(
        brownfield_map.to_json(),
        name="Brownfield Register 2025",
        style_function=lambda feature: {
            "fillColor": "#8b4513",
            "color": "#5a2d0c",
            "weight": 2,
            "fillOpacity": 0.40
        },
        tooltip="Birmingham Brownfield Register 2025"
    ).add_to(m)


# ------------------------------------------------------------
# LSOA RISK DOTS
# ------------------------------------------------------------

for _, row in indicators.iterrows():

    if row["Risk_Level"] == "HIGH":

        colour = "red"

    elif row["Risk_Level"] == "MODERATE":

        colour = "orange"

    else:

        colour = "green"

    popup_html = f"""
    <div style="width:270px">

    <h4>{row['LSOA21NM']}</h4>

    <b>Screening level:</b>
    {row['Risk_Level']}<br><br>

    <b>Official CRVA mean:</b>
    {row['CRVA_Mean']:.2f}<br>

    <b>Flood exposure:</b>
    {row['Flood_Exposure_Percent']:.2f}%<br>

    <b>Flood Zone 3:</b>
    {row['Flood_Zone_3_Percent']:.2f}%<br>

    <b>Surface flooding:</b>
    {row['Surface_Flood_Percent']:.2f}%<br>

    <b>Brownfield exposure:</b>
    {row['Brownfield_Percent']:.2f}%<br>

    <b>Brownfield sites:</b>
    {int(row['Brownfield_Site_Count'])}<br><br>

    <b>Engineering screening score:</b>
    {row['Screening_Percent']:.1f}%

    </div>
    """

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=10,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.90,
        weight=2,
        popup=folium.Popup(
            popup_html,
            max_width=350
        ),
        tooltip=(
            f"{row['LSOA21NM']} — "
            f"{row['Risk_Level']}"
        )
    ).add_to(m)


folium.LayerControl().add_to(m)


st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[]
)


# ============================================================
# MAP LEGEND
# ============================================================

st.markdown(
    """
    **Map key**

    🔴 **HIGH** — higher combined screening evidence

    🟠 **MODERATE** — moderate combined screening evidence

    🟢 **LOW** — lower combined screening evidence

    🔵 Flood Zone 3

    🟦 Surface-flooding evidence

    🟤 Brownfield Register 2025
    """
)


# ============================================================
# CRVA SECTION
# ============================================================

st.header(
    "Climate Risk & Vulnerability Assessment"
)

st.write(
    "The CRVA values shown below are Birmingham's official "
    "2025 CRVA results at 2021 LSOA level. They have not "
    "been artificially redistributed."
)

crva_table = indicators[
    [
        "LSOA21NM",
        "CRVA_Mean",
        "MINIMUM_RISK",
        "AVERAGE_RISK",
        "MAXIMUM_RISK"
    ]
].copy()

crva_table = crva_table.rename(
    columns={
        "LSOA21NM": "Ladywood statistical area",
        "CRVA_Mean": "Official CRVA mean",
        "MINIMUM_RISK": "Minimum risk",
        "AVERAGE_RISK": "Average risk",
        "MAXIMUM_RISK": "Maximum risk"
    }
)

st.dataframe(
    crva_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CRVA GRAPH
# ============================================================

fig_crva = px.bar(
    indicators.sort_values("CRVA_Mean"),
    x="CRVA_Mean",
    y="LSOA21NM",
    orientation="h",
    title="Official Birmingham CRVA Mean by Ladywood statistical area",
    labels={
        "CRVA_Mean": "CRVA mean",
        "LSOA21NM": "Ladywood statistical area"
    }
)

st.plotly_chart(
    fig_crva,
    use_container_width=True
)


# ============================================================
# FLOOD ANALYSIS
# ============================================================

st.header("Flood & Water Evidence")

flood_chart = indicators[
    [
        "LSOA21NM",
        "Flood_Zone_3_Percent",
        "Surface_Flood_Percent"
    ]
].copy()

flood_chart = flood_chart.rename(
    columns={
        "LSOA21NM": "Ladywood area",
        "Flood_Zone_3_Percent": "Flood Zone 3 (%)",
        "Surface_Flood_Percent": "Surface flooding (%)"
    }
)

melted_flood = flood_chart.melt(
    id_vars="Ladywood area",
    var_name="Evidence",
    value_name="Percentage"
)

fig_flood = px.bar(
    melted_flood,
    x="Ladywood area",
    y="Percentage",
    color="Evidence",
    barmode="group",
    title="Mapped flood exposure within Ladywood",
    labels={
        "Percentage": "Area affected (%)"
    }
)

st.plotly_chart(
    fig_flood,
    use_container_width=True
)


# ============================================================
# BROWNFIELD
# ============================================================

st.header("Brownfield Evidence")

brownfield_chart = indicators[
    [
        "LSOA21NM",
        "Brownfield_Percent",
        "Brownfield_Site_Count"
    ]
].copy()

brownfield_chart = brownfield_chart.rename(
    columns={
        "LSOA21NM": "Ladywood area",
        "Brownfield_Percent": "Brownfield coverage (%)",
        "Brownfield_Site_Count": "Brownfield sites"
    }
)

st.dataframe(
    brownfield_chart,
    use_container_width=True,
    hide_index=True
)

fig_brownfield = px.bar(
    indicators.sort_values(
        "Brownfield_Percent",
        ascending=False
    ),
    x="LSOA21NM",
    y="Brownfield_Percent",
    title="Brownfield coverage within Ladywood",
    labels={
        "LSOA21NM": "Ladywood statistical area",
        "Brownfield_Percent": "Brownfield coverage (%)"
    }
)

st.plotly_chart(
    fig_brownfield,
    use_container_width=True
)


# ============================================================
# OVERALL SCREENING
# ============================================================

st.header("Ladywood Environmental Screening Assessment")

st.write(
    "This screening assessment combines three measured "
    "spatial indicators: Birmingham CRVA, mapped flood "
    "exposure and mapped Brownfield Register exposure."
)

screen_table = indicators[
    [
        "LSOA21NM",
        "CRVA_Mean",
        "Flood_Exposure_Percent",
        "Brownfield_Percent",
        "Brownfield_Site_Count",
        "Screening_Percent",
        "Risk_Level"
    ]
].copy()

screen_table = screen_table.rename(
    columns={
        "LSOA21NM": "Ladywood area",
        "CRVA_Mean": "CRVA mean",
        "Flood_Exposure_Percent": "Flood exposure (%)",
        "Brownfield_Percent": "Brownfield coverage (%)",
        "Brownfield_Site_Count": "Brownfield sites",
        "Screening_Percent": "Screening score (%)",
        "Risk_Level": "Screening level"
    }
)

screen_table = screen_table.sort_values(
    "Screening score (%)",
    ascending=False
)

st.dataframe(
    screen_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# SCREENING GRAPH
# ============================================================

fig_screen = px.bar(
    indicators.sort_values(
        "Screening_Percent",
        ascending=True
    ),
    x="Screening_Percent",
    y="LSOA21NM",
    color="Risk_Level",
    orientation="h",
    title="Ladywood Environmental Screening Score",
    labels={
        "Screening_Percent": "Screening score (%)",
        "LSOA21NM": "Ladywood statistical area",
        "Risk_Level": "Screening level"
    }
)

st.plotly_chart(
    fig_screen,
    use_container_width=True
)


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================

st.header("Engineering Interpretation")

highest = indicators.sort_values(
    "Screening_Score",
    ascending=False
).iloc[0]

st.write(
    f"The area with the highest combined screening score is "
    f"**{highest['LSOA21NM']}**."
)

st.write(
    f"Its calculated screening score is "
    f"**{highest['Screening_Percent']:.1f}%**."
)

st.write(
    "The score should be interpreted as a project screening "
    "indicator. It is not an official Birmingham City Council "
    "risk classification."
)


# ============================================================
# DATA TABLE
# ============================================================

st.header("Ladywood Measurement Dataset")

display_columns = [
    "LSOA21NM",
    "LSOA_Area_ha",
    "CRVA_Mean",
    "Flood_Zone_3_Percent",
    "Surface_Flood_Percent",
    "Flood_Exposure_Percent",
    "Brownfield_Percent",
    "Brownfield_Site_Count",
    "Screening_Percent",
    "Risk_Level"
]

final_table = indicators[
    display_columns
].copy()

final_table = final_table.rename(
    columns={
        "LSOA21NM": "Ladywood area",
        "LSOA_Area_ha": "Area (ha)",
        "CRVA_Mean": "CRVA mean",
        "Flood_Zone_3_Percent": "Flood Zone 3 (%)",
        "Surface_Flood_Percent": "Surface flooding (%)",
        "Flood_Exposure_Percent": "Flood exposure (%)",
        "Brownfield_Percent": "Brownfield coverage (%)",
        "Brownfield_Site_Count": "Brownfield sites",
        "Screening_Percent": "Screening score (%)",
        "Risk_Level": "Screening level"
    }
)

st.dataframe(
    final_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = final_table.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Ladywood screening data",
    data=csv_data,
    file_name="Ladywood_environmental_screening.csv",
    mime="text/csv"
)


# ============================================================
# SOURCES
# ============================================================

st.header("Official Data Sources")

st.markdown(
    """
    **Birmingham City Council**

    • Climate Risk and Vulnerability Assessment (CRVA) 2025

    • CRVA by 2021 LSOA

    • Flood Risk Zone 3

    • Surface Flooding / Pluvial Flood Risk

    • Brownfield Register 2025

    • Birmingham Ward Boundaries

    The dashboard uses these sources directly through Birmingham
    City Council's ArcGIS services.
    """
)

st.caption(
    "All spatial calculations are restricted to the Ladywood "
    "ward boundary. Statistical areas are Birmingham 2021 LSOAs."
)
