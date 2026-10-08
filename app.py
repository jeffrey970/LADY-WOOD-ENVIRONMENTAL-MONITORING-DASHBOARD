import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import folium
import plotly.express as px
from streamlit_folium import st_folium
from shapely.geometry import Point
from shapely.ops import unary_union

st.set_page_config(
    page_title="LADYWOOD | Environmental Risk Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ---------------------------------------------------------
# OFFICIAL BIRMINGHAM CITY COUNCIL DATA
# ---------------------------------------------------------
CRVA_SERVICE = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"
PLANNING_SERVICE = "https://maps.birmingham.gov.uk/server/rest/services/planning/HELAA/MapServer"

CRVA_WARD = f"{CRVA_SERVICE}/12"
WARD_BOUNDARIES = f"{CRVA_SERVICE}/14"
FLUVIAL_Z3 = f"{CRVA_SERVICE}/109"
FLUVIAL_Z2 = f"{CRVA_SERVICE}/108"
PLUVIAL = f"{CRVA_SERVICE}/1546"
GREENSPACE = f"{CRVA_SERVICE}/1549"
WOODLAND = f"{CRVA_SERVICE}/1552"
BROWNFIELD = f"{PLANNING_SERVICE}/38"


@st.cache_data(ttl=3600, show_spinner=False)
def arcgis_query(url, where="1=1", bbox=None, fields="*", limit=2000):
    params = {
        "where": where,
        "outFields": fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": limit,
    }

    if bbox is not None:
        minx, miny, maxx, maxy = bbox
        params.update({
            "geometry": f"{minx},{miny},{maxx},{maxy}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })

    response = requests.get(url + "/query", params=params, timeout=45)
    response.raise_for_status()

    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"].get("message", "ArcGIS query failed."))

    features = data.get("features", [])
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return gdf


@st.cache_data(ttl=3600, show_spinner=False)
def get_ladywood():
    # CRVA By Ward already contains the official Ladywood geometry and
    # the official ward-level CRVA attributes.
    gdf = arcgis_query(
        CRVA_WARD,
        where="1=1",
        fields="WARDNME,WARD_CODE,MIN,MAX,MEAN,STD,MEDIAN,MINIMUM_RISK,AVERAGE_RISK,MAXIMUM_RISK",
    )
    if not gdf.empty and "WARDNME" in gdf.columns:
        ladywood = gdf[
            gdf["WARDNME"].astype(str).str.strip().str.casefold() == "ladywood"
        ].copy()
        if not ladywood.empty:
            return ladywood

    # Fallback to the official Ward Boundaries layer, again filtering locally.
    gdf = arcgis_query(
        WARD_BOUNDARIES,
        where="1=1",
        fields="WARDNME,WARD_CODE",
    )
    if not gdf.empty and "WARDNME" in gdf.columns:
        ladywood = gdf[
            gdf["WARDNME"].astype(str).str.strip().str.casefold() == "ladywood"
        ].copy()
        if not ladywood.empty:
            return ladywood

    raise RuntimeError(
        "Ladywood could not be identified in Birmingham's official ward services."
    )


def safe_layer(url, bbox, fields="*"):
    try:
        return arcgis_query(url, bbox=bbox, fields=fields)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def area_km2(gdf):
    if gdf is None or gdf.empty:
        return 0.0
    try:
        return float(gdf.to_crs(27700).geometry.area.sum() / 1_000_000)
    except Exception:
        return 0.0


def make_risk_points(boundary, z3, z2, pluvial, brownfield):
    """ Screening points are NOT invented pollution measurements. Each point is classified only from whether it falls inside official spatial evidence returned by Birmingham City Council. RED = high fluvial flood zone OR surface-flooding extent OR brownfield YELLOW = medium fluvial flood zone GREEN = none of those official spatial layers at that point """
    poly = unary_union(boundary.to_crs(4326).geometry)

    minx, miny, maxx, maxy = poly.bounds

    # Dense enough to visibly cover Ladywood while keeping the app fast.
    nx, ny = 11, 11
    points = []

    for x in np.linspace(minx, maxx, nx):
        for y in np.linspace(miny, maxy, ny):
            p = Point(x, y)
            if poly.contains(p):
                points.append(p)

    if not points:
        p = poly.centroid
        points = [p]

    def inside(gdf, point):
        if gdf is None or gdf.empty:
            return False
        try:
            return bool(gdf.geometry.contains(point).any())
        except Exception:
            return False

    records = []
    for p in points:
        high_flood = inside(z3, p)
        medium_flood = inside(z2, p)
        surface_flood = inside(pluvial, p)
        brown = inside(brownfield, p)

        if high_flood or surface_flood or brown:
            risk = "HIGH"
        elif medium_flood:
            risk = "MODERATE"
        else:
            risk = "LOW"

        evidence = []
        if high_flood:
            evidence.append("Flood Zone 3")
        if surface_flood:
            evidence.append("Surface flooding")
        if brown:
            evidence.append("Brownfield")
        if medium_flood:
            evidence.append("Flood Zone 2")
        if not evidence:
            evidence.append("No mapped screening evidence at point")

        records.append({
            "Latitude": p.y,
            "Longitude": p.x,
            "Risk": risk,
            "Evidence": ", ".join(evidence),
        })

    return pd.DataFrame(records)


def add_geojson(m, gdf, name, color, fill_opacity=0.25, weight=1):
    if gdf is None or gdf.empty:
        return

    folium.GeoJson(
        gdf.to_json(),
        name=name,
        style_function=lambda feature, c=color, fo=fill_opacity, w=weight: {
            "color": c,
            "weight": w,
            "fillColor": c,
            "fillOpacity": fo,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                f for f in gdf.columns
                if f not in ["geometry"] and gdf[f].dtype != "object" or
                (f in gdf.columns and f != "geometry")
            ][:4]
        ) if len(gdf.columns) > 1 else None,
    ).add_to(m)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
try:
    ladywood = get_ladywood()
except Exception as e:
    st.error(f"Ladywood data could not be loaded: {e}")
    st.stop()

ladywood = ladywood.to_crs(4326)
ladywood_poly = unary_union(ladywood.geometry)
centroid = ladywood_poly.centroid
map_center = [centroid.y, centroid.x]
bbox = tuple(ladywood_poly.bounds)

with st.spinner("Loading official Birmingham environmental data..."):
    zone3 = safe_layer(
        FLUVIAL_Z3,
        bbox,
        "origin,flood_zone,flood_sour,OBJECTID",
    )
    zone2 = safe_layer(
        FLUVIAL_Z2,
        bbox,
        "origin,flood_zone,flood_sour,OBJECTID",
    )
    pluvial = safe_layer(
        PLUVIAL,
        bbox,
        "pub_date,tile_id,OBJECTID",
    )
    greenspace = safe_layer(
        GREENSPACE,
        bbox,
        "type,OBJECTID",
    )
    woodland = safe_layer(
        WOODLAND,
        bbox,
        "type,OBJECTID",
    )
    brownfield = safe_layer(
        BROWNFIELD,
        bbox,
        "SiteReference,SiteNameAddress,Hectares,PlanningStatus,OBJECTID",
    )

risk_points = make_risk_points(ladywood, zone3, zone2, pluvial, brownfield)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("LADYWOOD")
st.subheader("Environmental Risk & Climate Dashboard")
st.caption("Birmingham, United Kingdom | Official spatial evidence screening")

st.info(
    "This dashboard uses published Birmingham City Council spatial datasets. "
    "The red/yellow/green points are an engineering screening visual based on "
    "mapped flood and brownfield evidence; they are not official Council risk ratings."
)

# ---------------------------------------------------------
# INDICATOR CARDS
# ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Ladywood area", f"{area_km2(ladywood):.2f} km²")
c2.metric("Flood Zone 3", f"{area_km2(zone3):.2f} km²")
c3.metric("Flood Zone 2", f"{area_km2(zone2):.2f} km²")
c4.metric("Surface flood extent", f"{area_km2(pluvial):.2f} km²")
c5.metric("Brownfield sites", str(len(brownfield)))

# ---------------------------------------------------------
# MAP
# ---------------------------------------------------------
st.header("Ladywood Environmental Risk Map")

m = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="OpenStreetMap",
    control_scale=True,
)

# Ladywood boundary
folium.GeoJson(
    ladywood.to_json(),
    name="Ladywood boundary",
    style_function=lambda feature: {
        "color": "#111111",
        "weight": 3,
        "fillColor": "#ffffff",
        "fillOpacity": 0.03,
    },
    tooltip="Ladywood ward",
).add_to(m)

# Official evidence layers
add_geojson(m, zone3, "Flood Zone 3 - high probability", "#e31a1c", 0.35, 1)
add_geojson(m, zone2, "Flood Zone 2 - medium probability", "#ffcc00", 0.25, 1)
add_geojson(m, pluvial, "Surface flooding", "#3182bd", 0.20, 1)
add_geojson(m, brownfield, "Brownfield Register 2025", "#8c510a", 0.35, 1)
add_geojson(m, greenspace, "Greenspace", "#31a354", 0.12, 1)
add_geojson(m, woodland, "Woodland", "#006d2c", 0.18, 1)

# Screening dots
colors = {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}

for _, row in risk_points.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=7,
        color=colors[row["Risk"]],
        fill=True,
        fill_color=colors[row["Risk"]],
        fill_opacity=0.9,
        weight=2,
        popup=folium.Popup(
            f"<b>Screening level:</b> {row['Risk']}<br>"
            f"<b>Evidence:</b> {row['Evidence']}",
            max_width=350,
        ),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=None, height=650, returned_objects=[])

st.caption(
    "Map layers: Birmingham City Council CRVA 2025, Birmingham Brownfield Register 2025. "
    "Screening dots are calculated from the spatial overlap of the official mapped layers."
)

# ---------------------------------------------------------
# RISK SUMMARY
# ---------------------------------------------------------
st.header("Environmental Screening Summary")

risk_counts = (
    risk_points["Risk"]
    .value_counts()
    .reindex(["HIGH", "MODERATE", "LOW"], fill_value=0)
    .reset_index()
)
risk_counts.columns = ["Risk level", "Number of screening points"]

left, right = st.columns(2)

with left:
    fig = px.bar(
        risk_counts,
        x="Risk level",
        y="Number of screening points",
        title="Ladywood screening points by risk level",
        text="Number of screening points",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    evidence_counts = (
        risk_points["Evidence"]
        .value_counts()
        .head(8)
        .reset_index()
    )
    evidence_counts.columns = ["Evidence", "Points"]
    fig2 = px.bar(
        evidence_counts,
        x="Points",
        y="Evidence",
        orientation="h",
        title="Mapped evidence behind the screening points",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# CLIMATE RISK / CRVA
# ---------------------------------------------------------
st.header("Climate Risk & Vulnerability")

crva_cols = [
    "WARDNME",
    "WARD_CODE",
    "MIN",
    "MAX",
    "MEAN",
    "STD",
    "MEDIAN",
    "MINIMUM_RISK",
    "AVERAGE_RISK",
    "MAXIMUM_RISK",
]

available = [c for c in crva_cols if c in ladywood.columns]

if "MEAN" in ladywood.columns:
    crva_mean = float(ladywood.iloc[0]["MEAN"])
    st.metric("Official Ladywood CRVA mean", f"{crva_mean:.2f}")

st.caption(
    "CRVA is reported at ward level by Birmingham City Council. "
    "It is deliberately NOT redistributed into the smaller screening points."
)

if available:
    st.dataframe(
        ladywood[available].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------
# FLOOD / WATER DATA
# ---------------------------------------------------------
st.header("Flood & Water Evidence")

flood_table = pd.DataFrame({
    "Dataset": [
        "Fluvial Flood Zone 3",
        "Fluvial Flood Zone 2",
        "Surface / Pluvial Flooding",
    ],
    "Mapped area in Ladywood (km²)": [
        area_km2(zone3),
        area_km2(zone2),
        area_km2(pluvial),
    ],
    "Features returned": [
        len(zone3),
        len(zone2),
        len(pluvial),
    ],
})

st.dataframe(flood_table, use_container_width=True, hide_index=True)

fig3 = px.bar(
    flood_table,
    x="Dataset",
    y="Mapped area in Ladywood (km²)",
    title="Mapped flood/water evidence within Ladywood",
)
st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------
# LAND / GREEN INFRASTRUCTURE
# ---------------------------------------------------------
st.header("Land, Brownfield & Green Infrastructure")

land_table = pd.DataFrame({
    "Dataset": ["Brownfield Register 2025", "Greenspace", "Woodland"],
    "Features in Ladywood": [len(brownfield), len(greenspace), len(woodland)],
    "Mapped area (km²)": [
        area_km2(brownfield),
        area_km2(greenspace),
        area_km2(woodland),
    ],
})

st.dataframe(land_table, use_container_width=True, hide_index=True)

fig4 = px.bar(
    land_table,
    x="Dataset",
    y="Mapped area (km²)",
    title="Land and green-infrastructure evidence",
)
st.plotly_chart(fig4, use_container_width=True)

# Brownfield details
if not brownfield.empty:
    wanted = [
        c for c in [
            "SiteReference",
            "SiteNameAddress",
            "Hectares",
            "PlanningStatus",
        ] if c in brownfield.columns
    ]
    if wanted:
        st.subheader("Brownfield Register sites within the Ladywood map extent")
        st.dataframe(
            brownfield[wanted].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------
# ENGINEERING INTERPRETATION
# ---------------------------------------------------------
st.header("Engineering Interpretation")

st.markdown(
    """ **HIGH screening points** identify locations where the mapped evidence includes Flood Zone 3, surface-flooding extent, or a mapped brownfield site. **MODERATE screening points** identify locations within the mapped Flood Zone 2 where no higher-priority mapped evidence was detected at the screening point. **LOW screening points** are locations where none of those mapped datasets intersected the screening point. These categories are a transparent student engineering screening method. They are not a replacement for a site-specific flood risk assessment, environmental assessment, or Birmingham City Council risk classification. """
)

# ---------------------------------------------------------
# SOURCES
# ---------------------------------------------------------
st.header("Official Data Sources")

st.markdown(
    """ - Birmingham City Council — Climate Risk and Vulnerability Assessment (CRVA) 2025 - Birmingham City Council — Flood Risk (Fluvial) Zone 2 and Zone 3 - Birmingham City Council — Risk of Surface Flooding (Pluvial) Extent - Birmingham City Council — Greenspace - Birmingham City Council — Woodland - Birmingham City Council — Brownfield Register 2025 """
)

# ---------------------------------------------------------
# DOWNLOAD DATA USED
# ---------------------------------------------------------
st.header("Download Screening Results")

csv = risk_points.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Ladywood screening points CSV",
    data=csv,
    file_name="Ladywood_environmental_screening_points.csv",
    mime="text/csv",
)
