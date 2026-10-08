import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import requests
import folium
import plotly.express as px
from io import StringIO
import streamlit.components.v1 as components


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Dashboard")


# ============================================================
# OFFICIAL BIRMINGHAM DATA
# ============================================================

BASE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_URL = f"{BASE}/14"
LSOA_URL = f"{BASE}/15"
FLOOD3_URL = f"{BASE}/109"
SURFACE_URL = f"{BASE}/1546"

BROWNFIELD_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROADS_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer/25"
)

AIR_PAGE = (
    "https://uk-air.defra.gov.uk/data/"
    "flat_files?site_id=BMLD"
)


# ============================================================
# GET ARCGIS DATA
# ============================================================

def get_layer(url, where="1=1"):

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson"
    }

    response = requests.get(
        url + "/query",
        params=params,
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:
        raise ValueError(
            "Birmingham GIS returned no features."
        )

    return gpd.GeoDataFrame.from_features(
        data["features"],
        crs="EPSG:4326"
    )


# ============================================================
# EXPOSURE CALCULATION
# ============================================================

def exposure_percentage(areas, evidence):

    a = areas.to_crs(27700)
    e = evidence.to_crs(27700)

    values = []

    for _, area in a.iterrows():

        total = area.geometry.area
        exposed = 0

        for _, item in e.iterrows():

            if area.geometry.intersects(
                item.geometry
            ):

                part = area.geometry.intersection(
                    item.geometry
                )

                exposed += part.area

        if total > 0:
            values.append(
                exposed / total * 100
            )
        else:
            values.append(0)

    return values


# ============================================================
# NORMALISATION
# ============================================================

def normalise(values):

    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    if values.dropna().empty:
        return pd.Series(
            0.0,
            index=values.index
        )

    low = values.min()
    high = values.max()

    if high == low:
        return pd.Series(
            0.0,
            index=values.index
        )

    return (
        values - low
    ) / (
        high - low
    )


def classify(value):

    if pd.isna(value):
        return "NO DATA"

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


# ============================================================
# LOAD OFFICIAL LADYWOOD DATA
# ============================================================

try:

    ladywood = get_layer(
        WARD_URL,
        "WARDNME = 'Ladywood'"
    )

    lsoa = get_layer(
        LSOA_URL
    )

    flood3 = get_layer(
        FLOOD3_URL
    )

    surface = get_layer(
        SURFACE_URL
    )

    brownfield = get_layer(
        BROWNFIELD_URL
    )

    roads = get_layer(
        ROADS_URL
    )

except Exception as error:

    st.error(
        "The official Birmingham data could not be loaded."
    )

    st.code(
        f"{type(error).__name__}: {error}"
    )

    st.stop()


# ============================================================
# CLIP LSOAs TO LADYWOOD
# ============================================================

ladywood_m = ladywood.to_crs(27700)
lsoa_m = lsoa.to_crs(27700)

ladywood_geometry = ladywood_m.geometry.unary_union

areas = []

for _, row in lsoa_m.iterrows():

    clipped = row.geometry.intersection(
        ladywood_geometry
    )

    if (
        not clipped.is_empty
        and clipped.area > 0
    ):

        new_row = row.drop(
            labels="geometry"
        ).to_dict()

        new_row["geometry"] = clipped

        areas.append(new_row)


areas = gpd.GeoDataFrame(
    areas,
    geometry="geometry",
    crs=27700
)

areas = areas.to_crs(4326)

areas["Area_ID"] = areas["LSOA21CD"]
areas["Area_Name"] = areas["LSOA21NM"]

areas["CRVA_Value"] = pd.to_numeric(
    areas["MEAN"],
    errors="coerce"
)


# ============================================================
# FLOOD EXPOSURE
# ============================================================

areas["Flood_Zone_3_pct"] = (
    exposure_percentage(
        areas,
        flood3
    )
)

areas["Surface_Flood_pct"] = (
    exposure_percentage(
        areas,
        surface
    )
)

# Prevent double-counting overlapping flood evidence.
areas["Flood_Exposure_pct"] = areas[
    [
        "Flood_Zone_3_pct",
        "Surface_Flood_pct"
    ]
].max(axis=1)


# ============================================================
# BROWNFIELD EXPOSURE
# ============================================================

areas["Brownfield_pct"] = (
    exposure_percentage(
        areas,
        brownfield
    )
)


# ============================================================
# ENGINEERING RISK MODEL
# ============================================================

areas["CRVA_Index"] = normalise(
    areas["CRVA_Value"]
)

areas["Flood_Index"] = normalise(
    areas["Flood_Exposure_pct"]
)

areas["Brownfield_Index"] = normalise(
    areas["Brownfield_pct"]
)

# Transparent equal-weight MCDA screening.
areas["Environmental_Index"] = (
    areas["CRVA_Index"]
    + areas["Flood_Index"]
    + areas["Brownfield_Index"]
) / 3


areas["Risk"] = (
    areas["Environmental_Index"]
    .apply(classify)
)


# ============================================================
# DOMINANT DRIVER
# ============================================================

def driver(row):

    scores = {
        "Climate vulnerability":
            row["CRVA_Index"],

        "Flood exposure":
            row["Flood_Index"],

        "Brownfield exposure":
            row["Brownfield_Index"]
    }

    return max(
        scores,
        key=scores.get
    )


areas["Dominant_Driver"] = areas.apply(
    driver,
    axis=1
)


# ============================================================
# ROAD AIR SCREENING
# ============================================================

areas_m = areas.to_crs(27700)
roads_m = roads.to_crs(27700)

traffic = []

for _, area in areas_m.iterrows():

    nearby = roads_m[
        roads_m.distance(
            area.geometry
        ) <= 100
    ]

    if "BRUM_CLASS" in nearby.columns:

        classes = (
            nearby["BRUM_CLASS"]
            .astype(str)
            .tolist()
        )

    else:

        classes = []

    road_text = " ".join(
        classes
    )

    if "A Road" in road_text:

        traffic.append("HIGH")

    elif "B Road" in road_text:

        traffic.append("MODERATE")

    else:

        traffic.append("LOW")


areas["Traffic_Air_Screening"] = traffic


# ============================================================
# DEFRA AIR QUALITY
# ============================================================

def load_air():

    try:

        page = requests.get(
            AIR_PAGE,
            timeout=60
        ).text

        import re

        links = re.findall(
            r'href=["\']([^"\']+\.csv[^"\']*)',
            page,
            flags=re.I
        )

        csv_url = None

        for link in links:

            if (
                "2025" in link
                and "BMLD" in link.upper()
            ):

                if link.startswith("/"):
                    csv_url = (
                        "https://uk-air.defra.gov.uk"
                        + link
                    )
                else:
                    csv_url = link

                break

        if csv_url is None:
            return pd.DataFrame()

        r = requests.get(
            csv_url,
            timeout=90
        )

        r.raise_for_status()

        text = r.content.decode(
            "utf-8",
            errors="replace"
        )

        try:

            df = pd.read_csv(
                StringIO(text),
                low_memory=False
            )

        except Exception:

            df = pd.read_csv(
                StringIO(text),
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
                if c.lower().strip()
                == "time"
            ),
            None
        )

        if date_col is None:
            return pd.DataFrame()

        if time_col:

            dt = pd.to_datetime(
                df[date_col].astype(str)
                + " "
                + df[time_col].astype(str),
                errors="coerce"
            )

        else:

            dt = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

        result = pd.DataFrame({
            "DateTime": dt
        })

        no2 = next(
            (
                c for c in columns
                if "no2" in c.lower()
            ),
            None
        )

        pm25 = next(
            (
                c for c in columns
                if (
                    "pm2.5" in c.lower()
                    or "pm25" in c.lower()
                )
            ),
            None
        )

        if no2:

            result["NO2"] = pd.to_numeric(
                df[no2],
                errors="coerce"
            )

        if pm25:

            result["PM2.5"] = pd.to_numeric(
                df[pm25],
                errors="coerce"
            )

        result = result[
            result["DateTime"]
            .dt.year
            .eq(2025)
        ]

        result["Month"] = (
            result["DateTime"]
            .dt.to_period("M")
            .astype(str)
        )

        return (
            result
            .groupby("Month")
            .mean(
                numeric_only=True
            )
            .reset_index()
        )

    except Exception:

        return pd.DataFrame()


air = load_air()


# ============================================================
# MAP
# ============================================================

centre = (
    ladywood
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


# Risk
risk_colours = {
    "HIGH": "#d73027",
    "MODERATE": "#fc8d59",
    "LOW": "#91cf60",
    "NO DATA": "#bdbdbd"
}

folium.GeoJson(
    areas.to_json(),
    name="Environmental risk",
    style_function=lambda feature: {
        "fillColor": risk_colours[
            feature["properties"]["Risk"]
        ],
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
            "Driver:"
        ]
    )
).add_to(m)


# Flood Zone 3
folium.GeoJson(
    flood3.to_json(),
    name="Flood Zone 3",
    style_function=lambda x: {
        "fillColor": "#2171b5",
        "color": "#08519c",
        "fillOpacity": 0.35
    }
).add_to(m)


# Surface flooding
folium.GeoJson(
    surface.to_json(),
    name="Surface flooding",
    style_function=lambda x: {
        "fillColor": "#6baed6",
        "color": "#2171b5",
        "fillOpacity": 0.30
    }
).add_to(m)


# Brownfield
folium.GeoJson(
    brownfield.to_json(),
    name="Brownfield",
    style_function=lambda x: {
        "fillColor": "#8c510a",
        "color": "#8c510a",
        "fillOpacity": 0.45
    }
).add_to(m)


# Roads
if "BRUM_CLASS" in roads.columns:

    classified = roads[
        roads["BRUM_CLASS"]
        .astype(str)
        .str.contains(
            "A Road|B Road|Classified",
            case=False,
            na=False
        )
    ]

else:

    classified = roads


folium.GeoJson(
    classified.to_json(),
    name="Classified roads",
    style_function=lambda x: {
        "color": "#555555",
        "weight": 3
    }
).add_to(m)


# Risk markers
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
            f"{row['Area_ID']}<br>"
            f"{row['Area_Name']}<br>"
            f"Risk: {row['Risk']}<br>"
            f"Index: "
            f"{row['Environmental_Index']:.2f}"
        )
    ).add_to(m)


folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# MAP DISPLAY
# ============================================================

st.subheader(
    "Ladywood Environmental Map"
)

# IMPORTANT:
# Render Folium directly as HTML.
# No streamlit-folium component is used.
components.html(
    m.get_root().render(),
    height=680,
    scrolling=False
)


# ============================================================
# RISK TABLE
# ============================================================

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
    ]
    .sort_values(
        "Environmental_Index",
        ascending=False
    )
    .reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# CRVA
# ============================================================

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
    title="CRVA by Ladywood area"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# FLOOD
# ============================================================

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


# ============================================================
# BROWNFIELD
# ============================================================

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


# ============================================================
# TRAFFIC
# ============================================================

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


# ============================================================
# MEASURED AIR QUALITY
# ============================================================

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
        "The 2025 Birmingham Ladywood air-quality file "
        "could not be read."
    )


# ============================================================
# ROADS
# ============================================================

st.subheader(
    "Classified Road Corridors Within Ladywood"
)

st.dataframe(
    classified.drop(
        columns="geometry",
        errors="ignore"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PRIORITY AREAS
# ============================================================

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


# ============================================================
# DOWNLOAD
# ============================================================

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
