import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import folium
import plotly.express as px
from io import StringIO
from streamlit_folium import st_folium

# =========================================================
# LADYWOOD ENVIRONMENTAL DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Dashboard")

# =========================================================
# OFFICIAL SOURCES
# =========================================================

CRVA = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"

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

AIR_FILE = (
    "https://uk-air.defra.gov.uk/datastore/data_files/"
    "site_data/BMLD_2025.csv"
)

# =========================================================
# GIS LOADER
# =========================================================

def get_layer(url, where="1=1"):

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson"
    }

    r = requests.get(
        url + "/query",
        params=params,
        timeout=60
    )

    r.raise_for_status()

    data = r.json()

    if "features" not in data:
        raise ValueError("No spatial features were returned.")

    return gpd.GeoDataFrame.from_features(
        data["features"],
        crs="EPSG:4326"
    )


# =========================================================
# ENGINEERING CALCULATIONS
# =========================================================

def exposure_percentage(areas, evidence):

    areas_m = areas.to_crs(27700)
    evidence_m = evidence.to_crs(27700)

    results = []

    for _, area in areas_m.iterrows():

        total_area = area.geometry.area
        exposed_area = 0

        for _, feature in evidence_m.iterrows():

            if area.geometry.intersects(feature.geometry):

                intersection = area.geometry.intersection(
                    feature.geometry
                )

                exposed_area += intersection.area

        if total_area > 0:
            results.append(
                exposed_area / total_area * 100
            )
        else:
            results.append(0)

    return results


def normalise(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    )

    if x.dropna().empty:
        return pd.Series(
            0,
            index=x.index
        )

    if x.max() == x.min():
        return pd.Series(
            0,
            index=x.index
        )

    return (x - x.min()) / (x.max() - x.min())


def risk_class(value):

    if pd.isna(value):
        return "NO DATA"

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


# =========================================================
# LOAD LADYWOOD
# =========================================================

try:

    ladywood = get_layer(
        WARD,
        "WARDNME = 'Ladywood'"
    )

    lsoa = get_layer(LSOA)

    flood3 = get_layer(FLOOD3)

    surface_flood = get_layer(
        SURFACE_FLOOD
    )

    brownfield = get_layer(
        BROWNFIELD
    )

    roads = get_layer(
        ROADS
    )

except Exception as e:

    st.error(
        "The official Birmingham spatial data could not be loaded."
    )

    st.code(
        f"{type(e).__name__}: {e}"
    )

    st.stop()


# =========================================================
# CLIP OFFICIAL LSOAs TO LADYWOOD
# =========================================================

ladywood_m = ladywood.to_crs(27700)
lsoa_m = lsoa.to_crs(27700)

ladywood_shape = ladywood_m.union_all()

areas = []

for _, row in lsoa_m.iterrows():

    clipped = row.geometry.intersection(
        ladywood_shape
    )

    if not clipped.is_empty and clipped.area > 0:

        new_row = row.drop(
            labels="geometry"
        ).to_dict()

        new_row["geometry"] = clipped

        areas.append(new_row)


areas = gpd.GeoDataFrame(
    areas,
    geometry="geometry",
    crs=27700
).to_crs(4326)


areas["Area_ID"] = areas["LSOA21CD"]
areas["Area_Name"] = areas["LSOA21NM"]


# =========================================================
# CRVA VALUE
# =========================================================

areas["CRVA_Value"] = np.nan

crva_m = lsoa_m = get_layer(
    LSOA
).to_crs(27700)

for i, area in areas.to_crs(27700).iterrows():

    matches = crva_m[
        crva_m.intersects(
            area.geometry
        )
    ]

    if not matches.empty:

        areas.loc[
            i,
            "CRVA_Value"
        ] = matches.iloc[0]["MEAN"]


# =========================================================
# FLOOD EXPOSURE
# =========================================================

areas["Flood_Zone_3_pct"] = exposure_percentage(
    areas,
    flood3
)

areas["Surface_Flood_pct"] = exposure_percentage(
    areas,
    surface_flood
)

# Use the larger extent so overlapping flood evidence
# is not counted twice.
areas["Flood_Exposure_pct"] = areas[
    [
        "Flood_Zone_3_pct",
        "Surface_Flood_pct"
    ]
].max(axis=1)


# =========================================================
# BROWNFIELD EXPOSURE
# =========================================================

areas["Brownfield_pct"] = exposure_percentage(
    areas,
    brownfield
)


# =========================================================
# NORMALISATION + COMPOSITE RISK
# =========================================================

areas["CRVA_Index"] = normalise(
    areas["CRVA_Value"]
)

areas["Flood_Index"] = normalise(
    areas["Flood_Exposure_pct"]
)

areas["Brownfield_Index"] = normalise(
    areas["Brownfield_pct"]
)

# Equal weighting gives each environmental stressor
# the same influence in the screening index.
areas["Environmental_Index"] = (
    areas["CRVA_Index"]
    + areas["Flood_Index"]
    + areas["Brownfield_Index"]
) / 3

areas["Risk"] = areas[
    "Environmental_Index"
].apply(risk_class)


# =========================================================
# DOMINANT ENVIRONMENTAL DRIVER
# =========================================================

def dominant_driver(row):

    values = {
        "Climate vulnerability":
            row["CRVA_Index"],

        "Flood exposure":
            row["Flood_Index"],

        "Brownfield exposure":
            row["Brownfield_Index"]
    }

    return max(
        values,
        key=values.get
    )


areas["Dominant_Driver"] = areas.apply(
    dominant_driver,
    axis=1
)


# =========================================================
# TRAFFIC AIR SCREENING
# =========================================================

areas_m = areas.to_crs(27700)
roads_m = roads.to_crs(27700)

traffic_result = []

for _, area in areas_m.iterrows():

    nearby = roads_m[
        roads_m.distance(
            area.geometry
        ) <= 100
    ]

    road_classes = (
        nearby["BRUM_CLASS"]
        .astype(str)
        .tolist()
        if "BRUM_CLASS" in nearby.columns
        else []
    )

    text = " ".join(
        road_classes
    )

    if "A Road" in text:
        traffic_result.append("HIGH")

    elif "B Road" in text:
        traffic_result.append("MODERATE")

    else:
        traffic_result.append("LOW")


areas["Traffic_Air_Screening"] = (
    traffic_result
)


# =========================================================
# 2025 LADYWOOD AIR QUALITY
# =========================================================

def load_air():

    try:

        r = requests.get(
            AIR_FILE,
            timeout=90
        )

        r.raise_for_status()

        raw = r.content.decode(
            "utf-8",
            errors="replace"
        )

        try:

            df = pd.read_csv(
                StringIO(raw),
                low_memory=False
            )

        except Exception:

            df = pd.read_csv(
                StringIO(raw),
                sep=None,
                engine="python"
            )

        columns = list(
            df.columns
        )

        date_col = next(
            (
                c for c in columns
                if "date" in c.lower()
                and "update" not in c.lower()
            ),
            None
        )

        time_col = next(
            (
                c for c in columns
                if c.lower().strip() == "time"
            ),
            None
        )

        datetime_col = next(
            (
                c for c in columns
                if "datetime" in c.lower()
                or "date/time" in c.lower()
            ),
            None
        )

        if datetime_col:

            dt = pd.to_datetime(
                df[datetime_col],
                errors="coerce"
            )

        elif date_col and time_col:

            dt = pd.to_datetime(
                df[date_col].astype(str)
                + " "
                + df[time_col].astype(str),
                errors="coerce"
            )

        elif date_col:

            dt = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

        else:

            return pd.DataFrame()

        result = pd.DataFrame({
            "DateTime": dt
        })

        no2_col = next(
            (
                c for c in columns
                if "no2" in c.lower()
            ),
            None
        )

        pm_col = next(
            (
                c for c in columns
                if "pm2.5" in c.lower()
                or "pm25" in c.lower()
            ),
            None
        )

        if no2_col:

            result["NO2"] = pd.to_numeric(
                df[no2_col],
                errors="coerce"
            )

        if pm_col:

            result["PM2.5"] = pd.to_numeric(
                df[pm_col],
                errors="coerce"
            )

        result = result[
            result["DateTime"].dt.year == 2025
        ]

        result["Month"] = (
            result["DateTime"]
            .dt.to_period("M")
            .astype(str)
        )

        return (
            result
            .groupby("Month")
            .mean(numeric_only=True)
            .reset_index()
        )

    except Exception:

        return pd.DataFrame()


air = load_air()


# =========================================================
# MAP
# =========================================================

centre = (
    ladywood
    .to_crs(4326)
    .geometry
    .union_all()
    .centroid
)

m = folium.Map(
    location=[
        centre.y,
        centre.x
    ],
    zoom_start=13,
    tiles="OpenStreetMap"
)


# Ladywood boundary
folium.GeoJson(
    ladywood.to_json(),
    name="Ladywood boundary",
    style_function=lambda x: {
        "fillOpacity": 0,
        "color": "black",
        "weight": 3
    }
).add_to(m)


# Official LSOAs
folium.GeoJson(
    areas.to_json(),
    name="Official LSOA boundaries",
    style_function=lambda x: {
        "fillOpacity": 0,
        "color": "#666666",
        "weight": 1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "Area_ID",
            "Area_Name"
        ],
        aliases=[
            "LSOA:",
            "Area:"
        ]
    )
).add_to(m)


# =========================================================
# RISK MAP
# =========================================================

risk_colours = {
    "HIGH": "#d73027",
    "MODERATE": "#fc8d59",
    "LOW": "#91cf60"
}

folium.GeoJson(
    areas.to_json(),
    name="Environmental risk",
    style_function=lambda feature: {
        "fillColor": risk_colours.get(
            feature["properties"]["Risk"],
            "#cccccc"
        ),
        "color": "black",
        "weight": 1,
        "fillOpacity": 0.55
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "Area_ID",
            "Area_Name",
            "Risk",
            "Environmental_Index",
            "Dominant_Driver"
        ],
        aliases=[
            "LSOA:",
            "Area:",
            "Risk:",
            "Index:",
            "Main driver:"
        ]
    )
).add_to(m)


# =========================================================
# FLOOD LAYERS
# =========================================================

folium.GeoJson(
    flood3.to_json(),
    name="Flood Zone 3",
    style_function=lambda x: {
        "fillColor": "#2171b5",
        "color": "#08519c",
        "fillOpacity": 0.35
    }
).add_to(m)

folium.GeoJson(
    surface_flood.to_json(),
    name="Surface flooding",
    style_function=lambda x: {
        "fillColor": "#6baed6",
        "color": "#2171b5",
        "fillOpacity": 0.30
    }
).add_to(m)


# =========================================================
# BROWNFIELD
# =========================================================

folium.GeoJson(
    brownfield.to_json(),
    name="Brownfield evidence",
    style_function=lambda x: {
        "fillColor": "#8c510a",
        "color": "#8c510a",
        "fillOpacity": 0.45
    }
).add_to(m)


# =========================================================
# CLASSIFIED ROADS
# =========================================================

if "BRUM_CLASS" in roads.columns:

    classified_roads = roads[
        roads["BRUM_CLASS"]
        .astype(str)
        .str.contains(
            "A Road|B Road|Classified",
            case=False,
            na=False
        )
    ]

else:

    classified_roads = roads


folium.GeoJson(
    classified_roads.to_json(),
    name="Classified roads",
    style_function=lambda x: {
        "color": "#555555",
        "weight": 3
    }
).add_to(m)


# =========================================================
# RISK MARKERS
# =========================================================

for _, row in areas.to_crs(4326).iterrows():

    point = row.geometry.representative_point()

    folium.CircleMarker(
        location=[
            point.y,
            point.x
        ],
        radius=5,
        color=risk_colours.get(
            row["Risk"],
            "#999999"
        ),
        fill=True,
        fill_opacity=0.9,
        popup=(
            f"<b>{row['Area_ID']}</b><br>"
            f"{row['Area_Name']}<br>"
            f"Risk: {row['Risk']}<br>"
            f"Index: "
            f"{row['Environmental_Index']:.2f}"
        )
    ).add_to(m)


folium.LayerControl(
    collapsed=False
).add_to(m)


# =========================================================
# DISPLAY MAP
# =========================================================

st.subheader(
    "Ladywood Environmental Map"
)

st_folium(
    m,
    height=650,
    use_container_width=True
)


# =========================================================
# RISK CLASSIFICATION
# =========================================================

st.subheader(
    "Environmental Screening Classification"
)

st.dataframe(
    areas[
        [
            "Area_ID",
            "Area_Name",
            "Risk",
            "Environmental_Index",
            "Dominant_Driver"
        ]
    ].sort_values(
        "Environmental_Index",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# CRVA
# =========================================================

st.subheader(
    "Climate Risk and Vulnerability"
)

fig = px.bar(
    areas.sort_values(
        "CRVA_Value",
        ascending=False
    ),
    x="Area_ID",
    y="CRVA_Value",
    hover_data=["Area_Name"],
    title="CRVA by Ladywood statistical area"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# FLOOD
# =========================================================

st.subheader(
    "Flood Exposure by Ladywood Area"
)

fig = px.bar(
    areas.sort_values(
        "Flood_Exposure_pct",
        ascending=False
    ),
    x="Area_ID",
    y="Flood_Exposure_pct",
    hover_data=[
        "Area_Name",
        "Flood_Zone_3_pct",
        "Surface_Flood_pct"
    ],
    labels={
        "Flood_Exposure_pct":
            "Exposed area (%)"
    },
    title="Combined flood exposure"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# BROWNFIELD
# =========================================================

st.subheader(
    "Brownfield Exposure by Ladywood Area"
)

fig = px.bar(
    areas.sort_values(
        "Brownfield_pct",
        ascending=False
    ),
    x="Area_ID",
    y="Brownfield_pct",
    hover_data=["Area_Name"],
    labels={
        "Brownfield_pct":
            "Brownfield area (%)"
    },
    title="Brownfield exposure"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# TRAFFIC
# =========================================================

st.subheader(
    "Traffic-related Air Pollution Screening"
)

st.dataframe(
    areas[
        [
            "Area_ID",
            "Area_Name",
            "Traffic_Air_Screening"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# MEASURED AIR QUALITY
# =========================================================

st.subheader(
    "Measured Air Quality — Birmingham Ladywood"
)

if not air.empty:

    pollutants = [
        x for x in [
            "NO2",
            "PM2.5"
        ]
        if x in air.columns
    ]

    if pollutants:

        pollutant = st.selectbox(
            "Pollutant",
            pollutants
        )

        fig = px.line(
            air.dropna(
                subset=[pollutant]
            ),
            x="Month",
            y=pollutant,
            markers=True,
            title=f"2025 {pollutant} monthly mean"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            air[
                ["Month"] + pollutants
            ],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "NO2 or PM2.5 was not found in the 2025 file."
        )

else:

    st.warning(
        "2025 Birmingham Ladywood air-quality data could not be read."
    )


# =========================================================
# CLASSIFIED ROADS
# =========================================================

st.subheader(
    "Classified Road Corridors Within Ladywood"
)

st.dataframe(
    classified_roads.drop(
        columns="geometry",
        errors="ignore"
    ),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PRIORITY AREAS
# =========================================================

st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

priority = areas.sort_values(
    "Environmental_Index",
    ascending=False
)

st.dataframe(
    priority[
        [
            "Area_ID",
            "Area_Name",
            "Risk",
            "Environmental_Index",
            "Dominant_Driver",
            "CRVA_Value",
            "Flood_Exposure_pct",
            "Brownfield_pct",
            "Traffic_Air_Screening"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD
# =========================================================

results = areas.drop(
    columns="geometry",
    errors="ignore"
)

st.download_button(
    "Download Ladywood Environmental Results",
    results.to_csv(
        index=False
    ).encode("utf-8"),
    "Ladywood_Environmental_Results.csv",
    "text/csv"
)
