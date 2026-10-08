# ============================================================
# LADYWOOD ENVIRONMENTAL MONITORING RISK DASHBOARD
# Wits Mining Engineering Project
# ============================================================

import io
import re
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import streamlit as st
import folium
import plotly.express as px

from streamlit_folium import st_folium


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Engineering screening settings
ROAD_DISTANCE = 100          # metres
HIGH = 0.67
MODERATE = 0.34

# Equal weighting avoids giving one environmental variable
# more importance simply because it uses a larger numerical scale.
CRVA_WEIGHT = 1 / 3
FLOOD_WEIGHT = 1 / 3
BROWNFIELD_WEIGHT = 1 / 3

# Main measured air-quality year
AIR_YEAR = 2025

# ============================================================
# 2. OFFICIAL BIRMINGHAM GIS SOURCES
# ============================================================

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD = f"{CRVA}/14"
LSOA = f"{CRVA}/15"
FLOOD3 = f"{CRVA}/109"
SURFACE_FLOOD = f"{CRVA}/1546"

BROWNFIELD = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROADS = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer/25"
)

ROAD_NAMES = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_HLC/MapServer/6"
)

DEFRA_PAGE = (
    "https://uk-air.defra.gov.uk/data/"
    "flat_files?site_id=BMLD"
)


# ============================================================
# 3. GENERAL GIS FUNCTION
# ============================================================

@st.cache_data(ttl=3600)
def get_layer(url, where="1=1"):

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": 4326
    }

    r = requests.get(
        f"{url}/query",
        params=params,
        timeout=45
    )

    r.raise_for_status()

    data = r.json()

    return gpd.GeoDataFrame.from_features(
        data["features"],
        crs="EPSG:4326"
    )


def metric(gdf):
    """British National Grid for metre/area calculations."""
    return gdf.to_crs(27700)


def clip(gdf, boundary):

    if gdf.empty:
        return gdf

    union = boundary.geometry.unary_union

    gdf = gdf[
        gdf.geometry.intersects(union)
    ].copy()

    if not gdf.empty:
        gdf["geometry"] = (
            gdf.geometry.intersection(union)
        )

    return gdf


def normalise(x):

    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    if x.max() == x.min():
        return pd.Series(
            0,
            index=x.index
        )

    return (
        (x - x.min())
        /
        (x.max() - x.min())
    )


def classify(x):

    if x >= HIGH:
        return "HIGH"

    if x >= MODERATE:
        return "MODERATE"

    return "LOW"


def colour(risk):

    return {
        "HIGH": "#d73027",
        "MODERATE": "#fdae61",
        "LOW": "#1a9850"
    }.get(
        risk,
        "#777777"
    )


# ============================================================
# 4. LADYWOOD BOUNDARY + OFFICIAL AREAS
# ============================================================

@st.cache_data(ttl=3600)
def load_boundary():

    ladywood = get_layer(
        WARD,
        "WARDNME = 'Ladywood'"
    )

    return metric(
        ladywood
    )


@st.cache_data(ttl=3600)
def load_lsoa(boundary):

    areas = metric(
        get_layer(LSOA)
    )

    return clip(
        areas,
        boundary
    )


# ============================================================
# 5. FLOOD / BROWNFIELD / ROAD DATA
# ============================================================

@st.cache_data(ttl=3600)
def load_flood3(boundary):

    return clip(
        metric(get_layer(FLOOD3)),
        boundary
    )


@st.cache_data(ttl=3600)
def load_surface_flood(boundary):

    return clip(
        metric(get_layer(SURFACE_FLOOD)),
        boundary
    )


@st.cache_data(ttl=3600)
def load_brownfield(boundary):

    return clip(
        metric(get_layer(BROWNFIELD)),
        boundary
    )


@st.cache_data(ttl=3600)
def load_roads(boundary):

    roads = clip(
        metric(get_layer(ROADS)),
        boundary
    )

    if roads.empty:
        return roads

    # Birmingham road classification field
    if "BRUM_CLASS" in roads.columns:
        roads["Classification"] = (
            roads["BRUM_CLASS"]
            .astype(str)
        )
    else:
        roads["Classification"] = "Classified road"

    roads = roads[
        roads["Classification"]
        .str.contains(
            "A Road|B Road|Classified",
            case=False,
            na=False
        )
    ]

    return roads


@st.cache_data(ttl=3600)
def load_road_names(boundary):

    try:
        names = clip(
            metric(get_layer(ROAD_NAMES)),
            boundary
        )
        return names
    except Exception:
        return gpd.GeoDataFrame(
            geometry=[],
            crs=27700
        )


# ============================================================
# 6. ENVIRONMENTAL ENGINEERING CALCULATIONS
# ============================================================

def exposure_percentage(areas, evidence):

    if evidence.empty:

        return pd.Series(
            0.0,
            index=areas.index
        )

    union = evidence.geometry.unary_union

    exposed = (
        areas.geometry
        .intersection(union)
        .area
    )

    total = areas.geometry.area

    return (
        exposed
        /
        total
        *
        100
    )


def calculate_results(
    areas,
    flood3,
    surface_flood,
    brownfield,
    roads
):

    result = areas.copy()

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    result["Area_m2"] = result.geometry.area

    # --------------------------------------------------------
    # CRVA
    # --------------------------------------------------------

    if "MEAN" in result.columns:

        result["CRVA"] = pd.to_numeric(
            result["MEAN"],
            errors="coerce"
        )

    else:

        result["CRVA"] = np.nan

    # --------------------------------------------------------
    # Flood exposure
    # --------------------------------------------------------

    result["Flood_Zone_3_%"] = (
        exposure_percentage(
            result,
            flood3
        )
    )

    result["Surface_Flood_%"] = (
        exposure_percentage(
            result,
            surface_flood
        )
    )

    # Do not add overlapping flood datasets.
    result["Flood_Exposure_%"] = result[
        [
            "Flood_Zone_3_%",
            "Surface_Flood_%"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # Brownfield exposure
    # --------------------------------------------------------

    result["Brownfield_%"] = (
        exposure_percentage(
            result,
            brownfield
        )
    )

    # --------------------------------------------------------
    # Normalise indicators to 0–1
    # --------------------------------------------------------

    result["CRVA_Index"] = normalise(
        result["CRVA"]
    )

    result["Flood_Index"] = normalise(
        result["Flood_Exposure_%"]
    )

    result["Brownfield_Index"] = normalise(
        result["Brownfield_%"]
    )

    # --------------------------------------------------------
    # Multi-criteria environmental screening
    # --------------------------------------------------------

    result["Environmental_Index"] = (
        result["CRVA_Index"]
        * CRVA_WEIGHT
        +
        result["Flood_Index"]
        * FLOOD_WEIGHT
        +
        result["Brownfield_Index"]
        * BROWNFIELD_WEIGHT
    )

    result["Environmental_%"] = (
        result["Environmental_Index"]
        * 100
    )

    result["Risk"] = (
        result["Environmental_Index"]
        .apply(classify)
    )

    # --------------------------------------------------------
    # Dominant environmental driver
    # --------------------------------------------------------

    result["Dominant_Driver"] = result[
        [
            "CRVA_Index",
            "Flood_Index",
            "Brownfield_Index"
        ]
    ].idxmax(axis=1).replace(
        {
            "CRVA_Index":
                "Climate vulnerability",
            "Flood_Index":
                "Flood exposure",
            "Brownfield_Index":
                "Brownfield / land disturbance"
        }
    )

    # --------------------------------------------------------
    # Traffic-related air screening
    # --------------------------------------------------------

    result["A_Road_100m"] = "No"
    result["B_Road_100m"] = "No"
    result["Road_Air_Risk"] = "LOW"

    if not roads.empty:

        for i, row in result.iterrows():

            nearby = roads[
                roads.geometry.distance(
                    row.geometry.centroid
                )
                <= ROAD_DISTANCE
            ]

            classes = (
                nearby["Classification"]
                .astype(str)
                .str.lower()
            )

            if classes.str.contains(
                "a road",
                na=False
            ).any():

                result.loc[
                    i,
                    "A_Road_100m"
                ] = "Yes"

                result.loc[
                    i,
                    "Road_Air_Risk"
                ] = "HIGH"

            elif classes.str.contains(
                "b road",
                na=False
            ).any():

                result.loc[
                    i,
                    "B_Road_100m"
                ] = "Yes"

                result.loc[
                    i,
                    "Road_Air_Risk"
                ] = "MODERATE"

    return result


# ============================================================
# 7. DEFRA LADYWOOD AIR QUALITY
# ============================================================

@st.cache_data(ttl=1800)
def load_air():

    # Official DEFRA Birmingham Ladywood annual file
    url = (
        "https://uk-air.defra.gov.uk/"
        f"datastore/data_files/site_data/"
        f"BMLD_{AIR_YEAR}.csv"
    )

    r = requests.get(
        url,
        timeout=45
    )

    r.raise_for_status()

    # DEFRA files can vary in delimiter.
    try:
        df = pd.read_csv(
            io.BytesIO(r.content),
            low_memory=False
        )
    except Exception:
        df = pd.read_csv(
            io.BytesIO(r.content),
            sep=";",
            low_memory=False
        )

    columns = [
        str(c).lower()
        for c in df.columns
    ]

    # Find date field
    date_col = next(
        (
            df.columns[i]
            for i, c in enumerate(columns)
            if "date" in c
            or "datetime" in c
        ),
        None
    )

    # Find NO2
    no2_col = next(
        (
            df.columns[i]
            for i, c in enumerate(columns)
            if "no2" in c
            or "nitrogen dioxide" in c
        ),
        None
    )

    # Find PM2.5
    pm_col = next(
        (
            df.columns[i]
            for i, c in enumerate(columns)
            if "pm2.5" in c
            or "pm25" in c
            or "pm 2.5" in c
        ),
        None
    )

    if date_col is None:
        raise ValueError(
            "DEFRA date field not found."
        )

    output = pd.DataFrame()

    output["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce",
        dayfirst=True
    )

    output["NO2"] = (
        pd.to_numeric(
            df[no2_col],
            errors="coerce"
        )
        if no2_col
        else np.nan
    )

    output["PM2.5"] = (
        pd.to_numeric(
            df[pm_col],
            errors="coerce"
        )
        if pm_col
        else np.nan
    )

    output = output.dropna(
        subset=["Date"]
    )

    output["Month"] = (
        output["Date"]
        .dt.strftime("%B")
    )

    return output


# ============================================================
# 8. ROAD NAMES
# ============================================================

def nearest_road_name(
    geometry,
    road_names
):

    if road_names.empty:
        return "Classified road corridor"

    if "ROAD_NAME" not in road_names.columns:
        return "Classified road corridor"

    distances = (
        road_names.geometry
        .distance(geometry)
    )

    index = distances.idxmin()

    if distances.loc[index] <= 30:

        name = road_names.loc[
            index,
            "ROAD_NAME"
        ]

        if pd.notna(name):
            return str(name)

    return "Classified road corridor"


# ============================================================
# 9. MAP
# ============================================================

def make_map(
    boundary,
    results,
    flood3,
    surface_flood,
    brownfield,
    roads
):

    centre = (
        boundary
        .to_crs(4326)
        .geometry
        .unary_union
        .centroid
    )

    m = folium.Map(
        location=[
            centre.y,
            centre.x
        ],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # --------------------------------------------------------
    # Ladywood boundary
    # --------------------------------------------------------

    folium.GeoJson(
        boundary.to_crs(4326).to_json(),
        name="Ladywood Boundary",
        style_function=lambda x: {
            "color": "black",
            "weight": 3,
            "fillOpacity": 0
        }
    ).add_to(m)

    # --------------------------------------------------------
    # Risk areas
    # --------------------------------------------------------

    risk_map = results.to_crs(4326)

    def risk_style(feature):

        risk = feature[
            "properties"
        ]["Risk"]

        return {
            "fillColor": colour(risk),
            "color": "#444444",
            "weight": 1,
            "fillOpacity": 0.35
        }

    folium.GeoJson(
        risk_map.to_json(),
        name="Environmental Screening",
        style_function=risk_style,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "Risk",
                "Environmental_%",
                "Dominant_Driver"
            ],
            aliases=[
                "Risk:",
                "Environmental index (%):",
                "Dominant driver:"
            ]
        )
    ).add_to(m)

    # --------------------------------------------------------
    # LSOA boundaries
    # --------------------------------------------------------

    folium.GeoJson(
        risk_map.to_json(),
        name="Official Ladywood Statistical Areas",
        style_function=lambda x: {
            "color": "#555555",
            "weight": 1,
            "dashArray": "5,5",
            "fillOpacity": 0
        }
    ).add_to(m)

    # --------------------------------------------------------
    # Flood Zone 3
    # --------------------------------------------------------

    if not flood3.empty:

        folium.GeoJson(
            flood3.to_crs(4326).to_json(),
            name="Flood Zone 3",
            style_function=lambda x: {
                "color": "#2166ac",
                "fillColor": "#67a9cf",
                "weight": 1,
                "fillOpacity": 0.30
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Surface flood
    # --------------------------------------------------------

    if not surface_flood.empty:

        folium.GeoJson(
            surface_flood
            .to_crs(4326)
            .to_json(),
            name="Surface Flood Evidence",
            style_function=lambda x: {
                "color": "#3182bd",
                "fillColor": "#9ecae1",
                "weight": 1,
                "fillOpacity": 0.25
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Brownfield
    # --------------------------------------------------------

    if not brownfield.empty:

        folium.GeoJson(
            brownfield.to_crs(4326).to_json(),
            name="Brownfield Evidence",
            style_function=lambda x: {
                "color": "#8c510a",
                "fillColor": "#d8b365",
                "weight": 1,
                "fillOpacity": 0.40
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Classified roads
    # --------------------------------------------------------

    if not roads.empty:

        folium.GeoJson(
            roads.to_crs(4326).to_json(),
            name="Classified Roads",
            style_function=lambda x: {
                "color": "#555555",
                "weight": 4
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Risk markers
    # --------------------------------------------------------

    for _, row in risk_map.iterrows():

        point = row.geometry.representative_point()

        folium.CircleMarker(
            location=[
                point.y,
                point.x
            ],
            radius=7,
            color=colour(
                row["Risk"]
            ),
            fill=True,
            fill_color=colour(
                row["Risk"]
            ),
            fill_opacity=1,
            popup=(
                f"Risk: {row['Risk']}<br>"
                f"Environmental index: "
                f"{row['Environmental_%']:.1f}%<br>"
                f"Dominant driver: "
                f"{row['Dominant_Driver']}"
            )
        ).add_to(m)

    # --------------------------------------------------------
    # Map key
    # --------------------------------------------------------

    legend = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 2px solid #555;
        font-size: 13px;
    ">
    <b>Ladywood Map Key</b><br><br>

    <span style="color:#d73027;">●</span>
    HIGH<br>

    <span style="color:#fdae61;">●</span>
    MODERATE<br>

    <span style="color:#1a9850;">●</span>
    LOW<br><br>

    <b>Map layers</b><br>
    Dark blue = Flood Zone 3<br>
    Light blue = Surface flood<br>
    Brown = Brownfield<br>
    Grey = Classified roads
    </div>
    """

    m.get_root().html.add_child(
        folium.Element(
            legend
        )
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# ============================================================
# 10. LOAD EVERYTHING
# ============================================================

boundary = load_boundary()
areas = load_lsoa(boundary)

flood3 = load_flood3(boundary)
surface_flood = load_surface_flood(boundary)
brownfield = load_brownfield(boundary)
roads = load_roads(boundary)
road_names = load_road_names(boundary)

results = calculate_results(
    areas,
    flood3,
    surface_flood,
    brownfield,
    roads
)

air = load_air()


# ============================================================
# 11. TITLE
# ============================================================

st.title(
    "Ladywood Environmental Monitoring Risk Dashboard"
)

st.caption(
    "Mining Engineering environmental screening | Ladywood, Birmingham"
)


# ============================================================
# 12. OVERVIEW
# ============================================================

st.subheader(
    "Ladywood Screening Overview"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Ladywood areas assessed",
    len(results)
)

c2.metric(
    "HIGH",
    (results["Risk"] == "HIGH").sum()
)

c3.metric(
    "MODERATE",
    (results["Risk"] == "MODERATE").sum()
)

c4.metric(
    "LOW",
    (results["Risk"] == "LOW").sum()
)


# ============================================================
# 13. MAP
# ============================================================

st.subheader(
    "Ladywood Environmental Risk Map"
)

st.write(
    "Use the layer control on the map to switch boundaries, "
    "flood evidence, brownfield evidence, roads and risk "
    "markers on or off."
)

m = make_map(
    boundary,
    results,
    flood3,
    surface_flood,
    brownfield,
    roads
)

st_folium(
    m,
    height=650,
    width=None
)


# ============================================================
# 14. ENVIRONMENTAL SCREENING
# ============================================================

st.subheader(
    "Environmental Screening Classification"
)

risk_counts = (
    results["Risk"]
    .value_counts()
    .reindex(
        ["HIGH", "MODERATE", "LOW"],
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk",
    "Areas"
]

fig = px.bar(
    risk_counts,
    x="Risk",
    y="Areas",
    text="Areas",
    title="Ladywood Environmental Screening"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 15. CRVA
# ============================================================

st.subheader(
    "Climate Risk and Vulnerability Assessment"
)

crva = (
    results[
        ["CRVA"]
    ]
    .sort_values(
        "CRVA",
        ascending=False
    )
)

crva["Ladywood Area"] = (
    "LSOA "
    + crva.index.astype(str)
)

fig = px.bar(
    crva,
    x="Ladywood Area",
    y="CRVA",
    title="CRVA by Ladywood Area"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 16. FLOOD EXPOSURE
# ============================================================

st.subheader(
    "Flood Exposure by Ladywood Area"
)

flood_table = results[
    [
        "Flood_Zone_3_%",
        "Surface_Flood_%",
        "Flood_Exposure_%"
    ]
].copy()

flood_table["Ladywood Area"] = (
    "LSOA "
    + flood_table.index.astype(str)
)

flood_long = flood_table.melt(
    id_vars="Ladywood Area",
    value_vars=[
        "Flood_Zone_3_%",
        "Surface_Flood_%"
    ],
    var_name="Flood evidence",
    value_name="Exposure (%)"
)

fig = px.bar(
    flood_long,
    x="Ladywood Area",
    y="Exposure (%)",
    color="Flood evidence",
    barmode="group",
    title="Flood Exposure by Ladywood Area"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 17. BROWNFIELD
# ============================================================

st.subheader(
    "Brownfield Exposure by Ladywood Area"
)

brown = results[
    ["Brownfield_%"]
].copy()

brown["Ladywood Area"] = (
    "LSOA "
    + brown.index.astype(str)
)

fig = px.bar(
    brown,
    x="Ladywood Area",
    y="Brownfield_%",
    title="Brownfield Exposure by Ladywood Area"
)

fig.update_layout(
    yaxis_title="Brownfield exposure (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# 18. TRAFFIC AIR SCREENING
# ============================================================

st.subheader(
    "Traffic-Related Air Pollution Screening"
)

st.caption(
    "This is a spatial source-proximity screen, not measured "
    "NO₂. The 100 m distance is a project screening assumption."
)

air_screen = results[
    [
        "A_Road_100m",
        "B_Road_100m",
        "Road_Air_Risk"
    ]
].copy()

air_screen["Ladywood Area"] = (
    "LSOA "
    + air_screen.index.astype(str)
)

air_screen = air_screen[
    [
        "Ladywood Area",
        "A_Road_100m",
        "B_Road_100m",
        "Road_Air_Risk"
    ]
]

st.dataframe(
    air_screen,
    use_container_width=True
)


# ============================================================
# 19. MEASURED LADYWOOD AIR QUALITY
# ============================================================

st.subheader(
    "Measured Air Quality — Birmingham Ladywood"
)

pollutant = st.selectbox(
    "Select measured pollutant",
    ["NO2", "PM2.5"]
)

if pollutant == "NO2":
    y = "NO2"
    title = "Birmingham Ladywood 2025 Monthly NO₂"
    ylabel = "NO₂ (µg/m³)"
else:
    y = "PM2.5"
    title = "Birmingham Ladywood 2025 Monthly PM₂.₅"
    ylabel = "PM₂.₅ (µg/m³)"

monthly = (
    air
    .set_index("Date")
    .resample("ME")[y]
    .mean()
    .reset_index()
)

fig = px.line(
    monthly,
    x="Date",
    y=y,
    markers=True,
    title=title
)

fig.update_layout(
    xaxis_title="Month",
    yaxis_title=ylabel
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "These are measured monitoring-station values from "
    "the Birmingham Ladywood DEFRA dataset. Missing "
    "measurements are not replaced with zero."
)


# ============================================================
# 20. CLASSIFIED ROAD CORRIDORS
# ============================================================

st.subheader(
    "Classified Road Corridors Within Ladywood"
)

road_rows = []

for _, road in roads.iterrows():

    road_rows.append(
        {
            "Road corridor":
                nearest_road_name(
                    road.geometry,
                    road_names
                ),
            "Classification":
                road["Classification"]
        }
    )

road_table = (
    pd.DataFrame(
        road_rows
    )
    .drop_duplicates()
)

st.dataframe(
    road_table,
    use_container_width=True
)


# ============================================================
# 21. PRIORITY AREAS
# ============================================================

st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

priority = results[
    [
        "Environmental_%",
        "Risk",
        "Dominant_Driver",
        "CRVA",
        "Flood_Exposure_%",
        "Brownfield_%",
        "Road_Air_Risk"
    ]
].copy()

priority = priority.sort_values(
    "Environmental_%",
    ascending=False
)

priority["Ladywood Area"] = (
    "LSOA "
    + priority.index.astype(str)
)

priority = priority[
    [
        "Ladywood Area",
        "Environmental_%",
        "Risk",
        "Dominant_Driver",
        "CRVA",
        "Flood_Exposure_%",
        "Brownfield_%",
        "Road_Air_Risk"
    ]
]

st.dataframe(
    priority.round(2),
    use_container_width=True
)


# ============================================================
# 22. DOWNLOAD RESULTS
# ============================================================

st.subheader(
    "Ladywood Environmental Results"
)

download = results.drop(
    columns="geometry"
).copy()

st.download_button(
    "Download Ladywood Environmental Results",
    download.to_csv(
        index=False
    ),
    "Ladywood_Environmental_Results.csv",
    "text/csv"
)
