import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import folium
import plotly.express as px

from streamlit_folium import st_folium
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
from folium.plugins import Fullscreen


# ============================================================
# LADYWOOD ENVIRONMENTAL RISK & CLIMATE DASHBOARD
# Official Birmingham spatial evidence
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Risk Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

ARCGIS = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = f"{ARCGIS}/14"

CRVA_WARD_LAYER = f"{ARCGIS}/12"

NO2_LAYER = f"{ARCGIS}/8"

PM25_LAYER = f"{ARCGIS}/9"

PLUVIAL_LAYER = f"{ARCGIS}/1546"

GREENSPACE_LAYER = f"{ARCGIS}/1549"

WOODLAND_LAYER = f"{ARCGIS}/1552"

TREE_CANOPY_LAYER = f"{ARCGIS}/3"

BROWNFIELD_API = (
    "https://www.planning.data.gov.uk/"
    "api/1.0/entity.json"
)

BIRMINGHAM_ORGANISATION = "44"


# ============================================================
# ADMIN SETTINGS
# ============================================================

if "admin_mode" not in st.session_state:
    st.session_state.admin_mode = False

if "refresh_data" not in st.session_state:
    st.session_state.refresh_data = False


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 44px;
        font-weight: 750;
        color: #17324d;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 18px;
        color: #66727c;
        margin-bottom: 25px;
    }

    .section {
        font-size: 26px;
        font-weight: 700;
        color: #17324d;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .info-box {
        background: #f5f7f9;
        border: 1px solid #d9dfe5;
        border-radius: 10px;
        padding: 18px;
    }

    .high {
        background: #f8d7da;
        border-left: 6px solid #c62828;
        padding: 14px;
        border-radius: 8px;
    }

    .moderate {
        background: #fff3cd;
        border-left: 6px solid #e0a800;
        padding: 14px;
        border-radius: 8px;
    }

    .low {
        background: #dff2e1;
        border-left: 6px solid #2e7d32;
        padding: 14px;
        border-radius: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GENERAL ARC GIS FUNCTION
# ============================================================

@st.cache_data(ttl=3600)
def get_feature_layer(url, where="1=1"):

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
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

    features = data.get("features", [])

    records = []

    for feature in features:

        geometry = feature.get("geometry")

        if geometry is None:
            continue

        properties = feature.get("properties", {}).copy()

        properties["geometry"] = shape(geometry)

        records.append(properties)

    if not records:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    return gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs="EPSG:4326"
    )


# ============================================================
# LADYWOOD BOUNDARY
# ============================================================

@st.cache_data(ttl=3600)
def get_ladywood():

    # Retrieve the official Birmingham ward layer first.
    # We deliberately do not assume the exact SQL spelling
    # of the ward name.

    wards = get_feature_layer(
        WARD_LAYER,
        "1=1"
    )

    if wards.empty:
        raise ValueError(
            "The official Birmingham ward boundary service "
            "returned no features."
        )

    # Find Ladywood from the returned official ward names.
    name_column = "WARDNME"

    if name_column not in wards.columns:
        raise ValueError(
            "The Birmingham ward layer does not contain "
            "the expected WARDNME field."
        )

    ladywood = wards[
        wards[name_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("ladywood")
    ].copy()

    # Fallback in case the published name contains
    # additional wording.
    if ladywood.empty:

        ladywood = wards[
            wards[name_column]
            .astype(str)
            .str.contains(
                "ladywood",
                case=False,
                na=False
            )
        ].copy()

    if ladywood.empty:

        available = sorted(
            wards[name_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "The Birmingham service loaded successfully, "
            "but Ladywood could not be identified. "
            f"Ward names returned: {available}"
        )

    return ladywood


# ============================================================
# BROWNFIELDS
# ============================================================

@st.cache_data(ttl=3600)
def get_brownfields():

    params = {
        "dataset": "brownfield-land",
        "organisation-entity": BIRMINGHAM_ORGANISATION,
        "limit": 5000
    }

    response = requests.get(
        BROWNFIELD_API,
        params=params,
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):

        records = data.get(
            "entities",
            data.get("results", [])
        )

    else:

        records = data

    rows = []

    for item in records:

        point_text = item.get("point")

        if not point_text:
            continue

        try:

            coords = point_text.replace(
                "POINT (",
                ""
            ).replace(
                ")",
                ""
            ).split()

            lon = float(coords[0])
            lat = float(coords[1])

        except Exception:
            continue

        rows.append(
            {
                "Reference":
                    item.get("reference"),

                "Site":
                    item.get("site-address"),

                "Hectares":
                    pd.to_numeric(
                        item.get("hectares"),
                        errors="coerce"
                    ),

                "Latitude":
                    lat,

                "Longitude":
                    lon,

                "Status":
                    item.get(
                        "planning-permission-status"
                    ),

                "Entry date":
                    item.get("entry-date")
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FILTER DATA TO LADYWOOD
# ============================================================

def filter_points_to_ladywood(
    df,
    ladywood
):

    if df.empty:
        return df

    polygon = unary_union(
        ladywood.geometry
    )

    points = [
        Point(
            row["Longitude"],
            row["Latitude"]
        )
        for _, row in df.iterrows()
    ]

    mask = [
        polygon.contains(point)
        for point in points
    ]

    return df.loc[mask].copy()


# ============================================================
# FLOOD DATA
# ============================================================

@st.cache_data(ttl=3600)
def get_flood_data():

    return get_feature_layer(
        PLUVIAL_LAYER,
        "1=1"
    )


# ============================================================
# GREENSPACE
# ============================================================

@st.cache_data(ttl=3600)
def get_greenspace():

    return get_feature_layer(
        GREENSPACE_LAYER,
        "1=1"
    )


# ============================================================
# WOODLAND
# ============================================================

@st.cache_data(ttl=3600)
def get_woodland():

    return get_feature_layer(
        WOODLAND_LAYER,
        "1=1"
    )


# ============================================================
# CRVA
# ============================================================

@st.cache_data(ttl=3600)
def get_crva():

    return get_feature_layer(
        CRVA_WARD_LAYER,
        "1=1"
    )


# ============================================================
# SAFE DATA LOADING
# ============================================================

try:

    ladywood = get_ladywood()

except Exception as error:

    st.error(
        f"Unable to load the official Ladywood boundary: {error}"
    )

    st.stop()


ladywood_polygon = unary_union(
    ladywood.geometry
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">LADYWOOD</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Environmental Risk & Climate Dashboard'
    ' — Birmingham, United Kingdom'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Dashboard controls"
)


# ------------------------------------------------------------
# ADMIN LOGIN
# ------------------------------------------------------------

with st.sidebar.expander(
    "Administrator"
):

    password = st.text_input(
        "Admin password",
        type="password"
    )

    if password:

        expected_password = st.secrets.get(
            "ADMIN_PASSWORD",
            "ladywood-admin"
        )

        if password == expected_password:

            st.session_state.admin_mode = True

        else:

            st.session_state.admin_mode = False

            st.error(
                "Incorrect password."
            )


# ============================================================
# PUBLIC CONTROLS
# ============================================================

st.sidebar.markdown(
    "### Map layers"
)

show_air = st.sidebar.checkbox(
    "Air pollution",
    True
)

show_flooding = st.sidebar.checkbox(
    "Surface-water flooding",
    True
)

show_brownfields = st.sidebar.checkbox(
    "Brownfield land",
    True
)

show_greenspace = st.sidebar.checkbox(
    "Greenspace",
    True
)

show_woodland = st.sidebar.checkbox(
    "Woodland",
    True
)


# ============================================================
# ADMIN CONTROLS
# ============================================================

if st.session_state.admin_mode:

    st.sidebar.success(
        "Administrator mode active"
    )

    st.sidebar.markdown(
        "### Analysis settings"
    )

    air_weight = st.sidebar.slider(
        "Air pollution weighting",
        0.0,
        1.0,
        0.35,
        0.05
    )

    flood_weight = st.sidebar.slider(
        "Flooding weighting",
        0.0,
        1.0,
        0.30,
        0.05
    )

    land_weight = st.sidebar.slider(
        "Land-condition weighting",
        0.0,
        1.0,
        0.20,
        0.05
    )

    green_weight = st.sidebar.slider(
        "Green-infrastructure weighting",
        0.0,
        1.0,
        0.15,
        0.05
    )

    st.sidebar.caption(
        "These settings affect the engineering screening "
        "index only. They do not alter official source data."
    )

    if st.sidebar.button(
        "Refresh official data"
    ):

        st.cache_data.clear()

        st.rerun()


# ============================================================
# MAP
# ============================================================

st.markdown(
    '<div class="section">Ladywood environmental map</div>',
    unsafe_allow_html=True
)


# Find map centre
centroid = ladywood.geometry.union_all().centroid

map_center = [
    centroid.y,
    centroid.x
]


m = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="CartoDB positron"
)


Fullscreen().add_to(m)


# ------------------------------------------------------------
# LADYWOOD BOUNDARY
# ------------------------------------------------------------

folium.GeoJson(
    ladywood.to_json(),
    name="Official Ladywood Ward Boundary",
    style_function=lambda feature: {
        "fillColor": "#ffffff",
        "color": "#17324d",
        "weight": 4,
        "fillOpacity": 0.05
    },
    tooltip=folium.GeoJsonTooltip(
        fields=["WARDNME"],
        aliases=["Ward:"]
    )
).add_to(m)


# ============================================================
# FLOODING
# ============================================================

flood_count = 0

try:

    flood = get_flood_data()

    if not flood.empty:

        flood = gpd.clip(
            flood,
            ladywood
        )

        flood_count = len(flood)

        if show_flooding:

            folium.GeoJson(
                flood.to_json(),
                name="Surface-water flood risk",
                style_function=lambda feature: {
                    "fillColor": "#4f81bd",
                    "color": "#24527a",
                    "weight": 1,
                    "fillOpacity": 0.35
                },
                tooltip="Official Birmingham surface-water flood-risk area"
            ).add_to(m)

except Exception as error:

    if st.session_state.admin_mode:

        st.warning(
            f"Flood layer unavailable: {error}"
        )


# ============================================================
# GREENSPACE
# ============================================================

greenspace_count = 0

try:

    greenspace = get_greenspace()

    greenspace = gpd.clip(
        greenspace,
        ladywood
    )

    greenspace_count = len(
        greenspace
    )

    if show_greenspace and not greenspace.empty:

        folium.GeoJson(
            greenspace.to_json(),
            name="Official greenspace",
            style_function=lambda feature: {
                "fillColor": "#63a85c",
                "color": "#3d7038",
                "weight": 1,
                "fillOpacity": 0.25
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["type"],
                aliases=["Greenspace type:"]
            )
        ).add_to(m)

except Exception as error:

    if st.session_state.admin_mode:

        st.warning(
            f"Greenspace layer unavailable: {error}"
        )


# ============================================================
# WOODLAND
# ============================================================

woodland_count = 0

try:

    woodland = get_woodland()

    woodland = gpd.clip(
        woodland,
        ladywood
    )

    woodland_count = len(
        woodland
    )

    if show_woodland and not woodland.empty:

        folium.GeoJson(
            woodland.to_json(),
            name="Official woodland",
            style_function=lambda feature: {
                "fillColor": "#26734d",
                "color": "#174d34",
                "weight": 1,
                "fillOpacity": 0.40
            }
        ).add_to(m)

except Exception as error:

    if st.session_state.admin_mode:

        st.warning(
            f"Woodland layer unavailable: {error}"
        )


# ============================================================
# BROWNFIELDS
# ============================================================

brownfield_data = pd.DataFrame()

try:

    brownfield_data = get_brownfields()

    brownfield_data = filter_points_to_ladywood(
        brownfield_data,
        ladywood
    )

    if show_brownfields:

        for _, site in brownfield_data.iterrows():

            popup = f"""
            <b>Brownfield site</b><br><br>
            Site: {site['Site']}<br>
            Reference: {site['Reference']}<br>
            Area: {site['Hectares']} ha<br>
            Planning status: {site['Status']}
            """

            folium.CircleMarker(
                location=[
                    site["Latitude"],
                    site["Longitude"]
                ],
                radius=6,
                color="#7b1fa2",
                fill=True,
                fill_color="#7b1fa2",
                fill_opacity=0.85,
                popup=folium.Popup(
                    popup,
                    max_width=320
                )
            ).add_to(m)

except Exception as error:

    if st.session_state.admin_mode:

        st.warning(
            f"Brownfield data unavailable: {error}"
        )


# ============================================================
# OFFICIAL AIR POLLUTION MAP
# ============================================================

# The Birmingham NO2 and PM2.5 layers are official raster
# datasets. They are displayed as spatial evidence rather
# than converted into invented point measurements.

if show_air:

    folium.raster_layers.WmsTileLayer(
        url=ARCGIS.replace(
            "/MapServer",
            "/MapServer/export"
        ),
        layers="show:8",
        fmt="image/png",
        transparent=True,
        name="Birmingham NO₂ spatial evidence",
        overlay=True,
        control=True,
        opacity=0.65
    ).add_to(m)


# ============================================================
# MAP LEGEND
# ============================================================

legend = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 14px;
border: 1px solid #999;
border-radius: 7px;
font-size: 13px;
">

<b>Environmental evidence</b><br><br>

<span style="color:#4f81bd">■</span>
Surface-water flood risk<br>

<span style="color:#7b1fa2">●</span>
Brownfield site<br>

<span style="color:#63a85c">■</span>
Greenspace<br>

<span style="color:#26734d">■</span>
Woodland<br>

<b>Boundary:</b>
Official Ladywood ward

</div>
"""

m.get_root().html.add_child(
    folium.Element(legend)
)


folium.LayerControl().add_to(m)


st_folium(
    m,
    width=None,
    height=650,
    returned_objects=[]
)


# ============================================================
# KEY INDICATORS
# ============================================================

st.markdown(
    '<div class="section">Ladywood environmental indicators</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Flood-risk features",
        flood_count
    )


with c2:

    st.metric(
        "Brownfield sites",
        len(brownfield_data)
    )


with c3:

    st.metric(
        "Greenspace features",
        greenspace_count
    )


with c4:

    st.metric(
        "Woodland features",
        woodland_count
    )


# ============================================================
# BROWNFIELD TABLE
# ============================================================

if not brownfield_data.empty:

    st.markdown(
        '<div class="section">Official brownfield evidence</div>',
        unsafe_allow_html=True
    )

    display_brownfields = brownfield_data[
        [
            "Reference",
            "Site",
            "Hectares",
            "Status"
        ]
    ].copy()

    display_brownfields = (
        display_brownfields
        .sort_values(
            "Hectares",
            ascending=False
        )
    )

    st.dataframe(
        display_brownfields,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CLIMATE RISK
# ============================================================

st.markdown(
    '<div class="section">Climate Risk and Vulnerability</div>',
    unsafe_allow_html=True
)


try:

    crva = get_crva()

    ladywood_crva = crva[
        crva.geometry.intersects(
            ladywood_polygon
        )
    ].copy()

    if not ladywood_crva.empty:

        row = ladywood_crva.iloc[0]

        values = {}

        for field in [
            "MEAN",
            "MEDIAN",
            "MIN",
            "MAX"
        ]:

            if field in row.index:

                values[field] = row[field]

        st.dataframe(
            pd.DataFrame(
                [values]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The CRVA value is ward-level evidence. "
            "It is not artificially redistributed between "
            "smaller Ladywood areas."
        )

except Exception as error:

    if st.session_state.admin_mode:

        st.warning(
            f"CRVA data unavailable: {error}"
        )


# ============================================================
# ENVIRONMENTAL EVIDENCE SUMMARY
# ============================================================

st.markdown(
    '<div class="section">Environmental evidence summary</div>',
    unsafe_allow_html=True
)


summary = pd.DataFrame(
    {
        "Indicator": [
            "Air pollution",
            "Surface-water flooding",
            "Brownfield land",
            "Greenspace",
            "Woodland/tree environment",
            "Climate vulnerability"
        ],

        "Official evidence": [
            "Birmingham CRVA NO₂ and PM2.5 spatial layers",
            "Birmingham CRVA surface-water flood layer",
            "UK Government Planning Data / Birmingham City Council",
            "Birmingham CRVA greenspace layer",
            "Birmingham CRVA woodland and vegetation data",
            "Birmingham CRVA"
        ],

        "Use in dashboard": [
            "Spatial pollution screening",
            "Flood exposure",
            "Land-condition evidence",
            "Green infrastructure context",
            "Tree/vegetation context",
            "Climate-risk context"
        ]
    }
)


st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# RISK SCREENING
# ============================================================

st.markdown(
    '<div class="section">Ladywood screening assessment</div>',
    unsafe_allow_html=True
)


st.write(
    """
    The dashboard does not claim that Birmingham City Council
    has officially divided Ladywood into environmental risk zones.

    Instead, screening areas are derived from the spatial
    distribution of official environmental evidence inside the
    Ladywood ward.

    This distinction is important: the result is an engineering
    screening tool, not an official Birmingham Council risk map.
    """
)


# ============================================================
# SIMPLE SPATIAL SCREENING GRID
# ============================================================

minx, miny, maxx, maxy = ladywood_polygon.bounds

grid_size = 0.002


cells = []

x_values = np.arange(
    minx,
    maxx,
    grid_size
)

y_values = np.arange(
    miny,
    maxy,
    grid_size
)


for x in x_values:

    for y in y_values:

        cell = box(
            x,
            y,
            x + grid_size,
            y + grid_size
        )

        if cell.intersects(
            ladywood_polygon
        ):

            cells.append(
                cell.intersection(
                    ladywood_polygon
                )
            )


grid = gpd.GeoDataFrame(
    {
        "Zone_ID": [
            f"LW-{i + 1:02d}"
            for i in range(len(cells))
        ]
    },
    geometry=cells,
    crs="EPSG:4326"
)


# ============================================================
# FLOOD EXPOSURE SCORE
# ============================================================

if not flood.empty:

    flood_union = unary_union(
        flood.geometry
    )

    grid["Flood_Area"] = grid.geometry.apply(
        lambda g:
        g.intersection(
            flood_union
        ).area
        if not g.is_empty
        else 0
    )

else:

    grid["Flood_Area"] = 0


# ============================================================
# BROWNFIELD PROXIMITY
# ============================================================

if not brownfield_data.empty:

    brownfield_points = [
        Point(
            row["Longitude"],
            row["Latitude"]
        )
        for _, row
        in brownfield_data.iterrows()
    ]

    grid["Brownfield_Count"] = grid.geometry.apply(
        lambda g:
        sum(
            g.contains(point)
            for point in brownfield_points
        )
    )

else:

    grid["Brownfield_Count"] = 0


# ============================================================
# GREENSPACE COVERAGE
# ============================================================

if not greenspace.empty:

    greenspace_union = unary_union(
        greenspace.geometry
    )

    grid["Greenspace_Area"] = grid.geometry.apply(
        lambda g:
        g.intersection(
            greenspace_union
        ).area
        if not g.is_empty
        else 0
    )

else:

    grid["Greenspace_Area"] = 0


# ============================================================
# NORMALISATION
# ============================================================

def normalise(series):

    maximum = series.max()

    minimum = series.min()

    if maximum == minimum:

        return pd.Series(
            np.zeros(len(series)),
            index=series.index
        )

    return (
        (series - minimum)
        /
        (maximum - minimum)
    )


grid["Flood_Index"] = normalise(
    grid["Flood_Area"]
)

grid["Brownfield_Index"] = normalise(
    grid["Brownfield_Count"]
)

grid["Green_Deficit_Index"] = (
    1 -
    normalise(
        grid["Greenspace_Area"]
    )
)


# ============================================================
# SCREENING SCORE
# ============================================================

# Air is not assigned a fake numerical value here.
# The official NO2/PM2.5 raster remains a spatial evidence
# layer. Flooding and land-condition calculations are derived
# directly from official vector data.

grid["Screening_Index"] = (

    grid["Flood_Index"] *
    flood_weight

    +

    grid["Brownfield_Index"] *
    land_weight

    +

    grid["Green_Deficit_Index"] *
    green_weight

)


# If public mode is used, weights are fixed.
# Admin mode allows controlled experimentation.

if not st.session_state.admin_mode:

    total = (
        0.30 +
        0.20 +
        0.15
    )

else:

    total = (
        flood_weight +
        land_weight +
        green_weight
    )


if total > 0:

    grid["Screening_Index"] = (
        grid["Screening_Index"] /
        total
    )


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify(value):

    if value >= 0.67:

        return "HIGH"

    if value >= 0.34:

        return "MODERATE"

    return "LOW"


grid["Risk_Class"] = (
    grid["Screening_Index"]
    .apply(classify)
)


# ============================================================
# SCREENING MAP
# ============================================================

st.markdown(
    '<div class="section">Environmental screening zones</div>',
    unsafe_allow_html=True
)


def grid_colour(feature):

    risk = feature[
        "properties"
    ]["Risk_Class"]

    if risk == "HIGH":

        return "#c62828"

    if risk == "MODERATE":

        return "#f0a500"

    return "#2e7d32"


screen_map = folium.Map(
    location=map_center,
    zoom_start=13,
    tiles="CartoDB positron"
)


folium.GeoJson(
    grid.to_json(),
    name="Ladywood screening zones",
    style_function=lambda feature: {
        "fillColor":
            grid_colour(feature),
        "color":
            "#555555",
        "weight":
            1,
        "fillOpacity":
            0.35
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "Zone_ID",
            "Risk_Class",
            "Screening_Index"
        ],
        aliases=[
            "Screening zone:",
            "Risk:",
            "Index:"
        ]
    )
).add_to(screen_map)


folium.GeoJson(
    ladywood.to_json(),
    name="Ladywood boundary",
    style_function=lambda feature: {
        "fillColor": "transparent",
        "color": "#17324d",
        "weight": 4,
        "fillOpacity": 0
    }
).add_to(screen_map)


folium.LayerControl().add_to(
    screen_map
)


st_folium(
    screen_map,
    width=None,
    height=600,
    returned_objects=[]
)


# ============================================================
# RISK CHART
# ============================================================

risk_counts = (
    grid["Risk_Class"]
    .value_counts()
    .reindex(
        [
            "HIGH",
            "MODERATE",
            "LOW"
        ],
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk",
    "Number of screening cells"
]


fig = px.bar(
    risk_counts,
    x="Risk",
    y="Number of screening cells",
    title="Ladywood environmental screening classification"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# HIGHEST SCREENING AREAS
# ============================================================

top_zones = (
    grid[
        [
            "Zone_ID",
            "Flood_Index",
            "Brownfield_Index",
            "Green_Deficit_Index",
            "Screening_Index",
            "Risk_Class"
        ]
    ]
    .sort_values(
        "Screening_Index",
        ascending=False
    )
    .head(10)
)


st.markdown(
    '<div class="section">Highest screening areas</div>',
    unsafe_allow_html=True
)


st.dataframe(
    top_zones,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ADMIN DATA VIEW
# ============================================================

if st.session_state.admin_mode:

    st.markdown(
        '<div class="section">Administrator data inspection</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This section is visible only when administrator mode "
        "is enabled."
    )

    st.write(
        "Ladywood boundary records:",
        len(ladywood)
    )

    st.write(
        "Brownfield records inside Ladywood:",
        len(brownfield_data)
    )

    st.write(
        "Flood polygons intersecting Ladywood:",
        flood_count
    )

    st.write(
        "Greenspace polygons intersecting Ladywood:",
        greenspace_count
    )

    st.write(
        "Woodland polygons intersecting Ladywood:",
        woodland_count
    )

    st.download_button(
        "Download screening-zone results",
        data=grid.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="Ladywood_environmental_screening.csv",
        mime="text/csv"
    )


# ============================================================
# DATA LIMITATIONS
# ============================================================

st.markdown(
    '<div class="section">How to interpret this dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

    <b>Measured / official spatial evidence</b><br>
    Birmingham's published environmental GIS layers are used
    directly where available.

    <br><br>

    <b>Screening analysis</b><br>
    The screening index is calculated by this dashboard from
    official spatial features. It is not an official Birmingham
    City Council risk score.

    <br><br>

    <b>Air pollution</b><br>
    The Birmingham NO₂ and PM2.5 layers are spatial datasets.
    The dashboard therefore displays them as spatial evidence
    rather than creating artificial four-zone measurements.

    <br><br>

    <b>Brownfield land</b><br>
    Brownfield locations are retrieved from the UK Government
    Planning Data service and restricted to sites falling
    inside the Ladywood boundary.

    <br><br>

    <b>Flooding</b><br>
    Flood polygons represent mapped flood-risk exposure. They
    should not be interpreted as a prediction of exact flood
    depth at a building.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SOURCES
# ============================================================

st.markdown(
    '<div class="section">Primary data sources</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    **Birmingham City Council — CRVA 2025 GIS**

    Birmingham City Council spatial data covering ward boundaries,
    climate vulnerability, NO₂, PM2.5, flooding, greenspace,
    woodland and vegetation/tree-canopy information.

    **Birmingham City Council — Flood Risk Assessments**

    Official Strategic Flood Risk Assessment material and
    Ladywood map appendices.

    **UK Government Planning Data — Brownfield land**

    Brownfield records supplied by Birmingham City Council.

    **Birmingham City Council — Air Quality**

    Birmingham air-quality monitoring and modelling information.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Ladywood Environmental Risk & Climate Dashboard | "
    "Wits Mining Engineering project | "
    "Primary-source environmental screening"
)
