import streamlit as st
import pandas as pd
import requests
import folium

from io import BytesIO
from shapely.geometry import shape, Polygon, MultiPolygon, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer


# ============================================================
# LADYWOOD ENVIRONMENTAL MONITORING DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham, UK | Environmental screening using official Birmingham City Council and DEFRA data")


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = CRVA + "/14"
LSOA_LAYER = CRVA + "/17"
FLOOD_ZONE_3_LAYER = CRVA + "/109"
SURFACE_FLOOD_LAYER = CRVA + "/1546"

BROWNFIELD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROAD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer/25"
)

AIR_2025_URL = (
    "https://uk-air.defra.gov.uk/datastore/data_files/"
    "site_data/BMLD_2025.csv?v=1"
)


# ============================================================
# COORDINATE SYSTEMS
# ============================================================

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

to_wgs84 = Transformer.from_crs(
    BNG,
    WGS84,
    always_xy=True
).transform


# ============================================================
# ARC GIS FUNCTIONS
# ============================================================

def request_json(url, params):
    """Request ArcGIS JSON safely."""

    try:
        response = requests.get(
            url,
            params=params,
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        if "error" in data:
            return None, str(data["error"])

        return data, None

    except Exception as e:
        return None, str(e)


def esri_geometry_to_shape(geometry):
    """
    Convert an ArcGIS ESRI polygon geometry into
    a Shapely geometry.

    ArcGIS polygon geometry contains rings.
    """

    if not geometry:
        return None

    rings = geometry.get("rings")

    if not rings:
        return None

    polygons = []

    for ring in rings:

        if len(ring) < 4:
            continue

        try:
            polygon = Polygon(ring)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if not polygon.is_empty:
                polygons.append(polygon)

        except Exception:
            continue

    if not polygons:
        return None

    if len(polygons) == 1:
        return polygons[0]

    return unary_union(polygons)


def get_features(
    layer_url,
    where="1=1",
    out_fields="*",
    geometry=None
):
    """
    Retrieve ArcGIS features using native ArcGIS JSON.

    This deliberately avoids the GeoJSON endpoint that caused
    the previous Ladywood-boundary problem.
    """

    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "27700",
        "f": "json"
    }

    if geometry is not None:

        params["geometry"] = ",".join(
            str(round(value, 3))
            for value in geometry
        )

        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"

    data, error = request_json(
        layer_url + "/query",
        params
    )

    if data is None:
        raise RuntimeError(error)

    features = data.get("features", [])

    rows = []

    for feature in features:

        attributes = feature.get("attributes", {}).copy()

        geometry_data = feature.get("geometry")

        geom = esri_geometry_to_shape(
            geometry_data
        )

        attributes["geometry"] = geom

        rows.append(attributes)

    return rows


# ============================================================
# LOAD THE OFFICIAL LADYWOOD BOUNDARY
# ============================================================

st.subheader("Ladywood boundary")

ladywood_rows = None
boundary_error = None


# First method: exact official ward name.
try:

    ladywood_rows = get_features(
        WARD_LAYER,
        where="WARDNME = 'Ladywood'"
    )

except Exception as e:

    boundary_error = str(e)


# Second method:
# retrieve ward features and identify Ladywood locally.
# This prevents the dashboard depending on the SQL filter.
if not ladywood_rows:

    try:

        all_wards = get_features(
            WARD_LAYER,
            where="1=1"
        )

        for row in all_wards:

            name = str(
                row.get("WARDNME", "")
            ).strip().lower()

            if name == "ladywood":
                ladywood_rows = [row]
                break

    except Exception as e:

        boundary_error = str(e)


if not ladywood_rows:

    st.error(
        "The official Birmingham Ladywood boundary could not be retrieved."
    )

    st.write(
        "The dashboard stopped because it will not substitute "
        "an invented boundary for the official Ladywood ward."
    )

    if boundary_error:
        st.caption(boundary_error)

    st.stop()


ladywood_geometry = ladywood_rows[0]["geometry"]


if ladywood_geometry is None or ladywood_geometry.is_empty:

    st.error(
        "The official Ladywood record was found, but its polygon geometry was empty."
    )

    st.stop()


ladywood_geometry = ladywood_geometry.buffer(0)


# ============================================================
# LADYWOOD EXTENT
# ============================================================

minx, miny, maxx, maxy = ladywood_geometry.bounds

ladywood_bbox = (
    minx,
    miny,
    maxx,
    maxy
)


# ============================================================
# LOAD OFFICIAL LSOAs
# ============================================================

try:

    lsoa_rows = get_features(
        LSOA_LAYER,
        where="1=1",
        geometry=ladywood_bbox
    )

except Exception as e:

    st.error("The official LSOA data could not be loaded.")

    st.caption(str(e))

    st.stop()


lsoa_records = []


for row in lsoa_rows:

    geom = row.get("geometry")

    if geom is None or geom.is_empty:
        continue

    # Keep only the portion inside Ladywood.
    clipped = geom.intersection(
        ladywood_geometry
    )

    if clipped.is_empty:
        continue

    area = clipped.area

    if area <= 0:
        continue

    lsoa_records.append(
        {
            "LSOA21CD": row.get("LSOA21CD"),
            "LSOA21NM": row.get("LSOA21NM"),
            "geometry": clipped,
            "Area_m2": area
        }
    )


lsoa_df = pd.DataFrame(
    lsoa_records
)


if lsoa_df.empty:

    st.error(
        "No official LSOA areas were found inside Ladywood."
    )

    st.stop()


# ============================================================
# GENERIC SPATIAL LOADER
# ============================================================

def load_spatial_layer(layer_url):

    try:

        rows = get_features(
            layer_url,
            where="1=1",
            geometry=ladywood_bbox
        )

    except Exception:

        return []

    result = []

    for row in rows:

        geom = row.get("geometry")

        if geom is None or geom.is_empty:
            continue

        clipped = geom.intersection(
            ladywood_geometry
        )

        if clipped.is_empty:
            continue

        result.append(
            {
                "attributes": row,
                "geometry": clipped
            }
        )

    return result


# ============================================================
# FLOOD ZONE 3
# ============================================================

flood3_features = load_spatial_layer(
    FLOOD_ZONE_3_LAYER
)


# ============================================================
# SURFACE FLOODING
# ============================================================

surface_flood_features = load_spatial_layer(
    SURFACE_FLOOD_LAYER
)


# ============================================================
# BROWNFIELD REGISTER
# ============================================================

brownfield_features = load_spatial_layer(
    BROWNFIELD_LAYER
)


# ============================================================
# ROADS
# ============================================================

road_features = load_spatial_layer(
    ROAD_LAYER
)


# ============================================================
# EXPOSURE CALCULATIONS
# ============================================================

def exposure_by_lsoa(
    lsoa_geometry,
    environmental_features
):

    if not environmental_features:
        return 0.0

    exposed_geometry = []

    for item in environmental_features:

        geom = item["geometry"]

        if geom is None or geom.is_empty:
            continue

        if geom.intersects(lsoa_geometry):

            intersection = geom.intersection(
                lsoa_geometry
            )

            if not intersection.is_empty:
                exposed_geometry.append(
                    intersection
                )

    if not exposed_geometry:
        return 0.0

    combined = unary_union(
        exposed_geometry
    )

    exposed_area = combined.area

    total_area = lsoa_geometry.area

    if total_area == 0:
        return 0.0

    return (
        exposed_area /
        total_area
    ) * 100


# ============================================================
# BUILD ENVIRONMENTAL TABLE
# ============================================================

results = []


for _, lsoa in lsoa_df.iterrows():

    geom = lsoa["geometry"]

    flood3_pct = exposure_by_lsoa(
        geom,
        flood3_features
    )

    surface_pct = exposure_by_lsoa(
        geom,
        surface_flood_features
    )

    brownfield_pct = exposure_by_lsoa(
        geom,
        brownfield_features
    )

    # Avoid double-counting overlapping flood categories.
    flood_pct = max(
        flood3_pct,
        surface_pct
    )

    results.append(
        {
            "LSOA21CD": lsoa["LSOA21CD"],
            "LSOA21NM": lsoa["LSOA21NM"],
            "Area_m2": lsoa["Area_m2"],
            "Flood_Zone_3_%": flood3_pct,
            "Surface_Flood_%": surface_pct,
            "Combined_Flood_Exposure_%": flood_pct,
            "Brownfield_Exposure_%": brownfield_pct,
            "geometry": geom
        }
    )


risk_df = pd.DataFrame(
    results
)


# ============================================================
# NORMALISATION
# ============================================================

def min_max(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(
            [0.0] * len(series),
            index=series.index
        )

    return (
        series - minimum
    ) / (
        maximum - minimum
    )


risk_df["Flood_Index"] = min_max(
    risk_df["Combined_Flood_Exposure_%"]
)

risk_df["Brownfield_Index"] = min_max(
    risk_df["Brownfield_Exposure_%"]
)


# ============================================================
# CRVA RISK
# ============================================================

crva_values = []

for _, row in risk_df.iterrows():

    matches = lsoa_df[
        lsoa_df["LSOA21CD"] ==
        row["LSOA21CD"]
    ]

    if matches.empty:

        crva_values.append(0.0)

        continue

    # Use the official CRVA average-risk field
    # when available.
    original = next(
        (
            x for x in lsoa_rows
            if x.get("LSOA21CD") ==
            row["LSOA21CD"]
        ),
        None
    )

    if original is None:

        crva_values.append(0.0)

    else:

        value = original.get(
            "AVERAGE_RISK"
        )

        try:
            crva_values.append(
                float(value)
            )
        except:
            crva_values.append(0.0)


risk_df["CRVA_Average_Risk"] = (
    crva_values
)

risk_df["CRVA_Index"] = min_max(
    risk_df["CRVA_Average_Risk"]
)


# ============================================================
# ENGINEERING ENVIRONMENTAL INDEX
# ============================================================

risk_df["Environmental_Index"] = (
    risk_df["CRVA_Index"] +
    risk_df["Flood_Index"] +
    risk_df["Brownfield_Index"]
) / 3


def classify_risk(value):

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


risk_df["Risk_Class"] = (
    risk_df["Environmental_Index"]
    .apply(classify_risk)
)


# ============================================================
# DOMINANT ENVIRONMENTAL DRIVER
# ============================================================

def dominant_driver(row):

    drivers = {
        "Climate vulnerability": row["CRVA_Index"],
        "Flood exposure": row["Flood_Index"],
        "Brownfield exposure": row["Brownfield_Index"]
    }

    return max(
        drivers,
        key=drivers.get
    )


risk_df["Dominant_Driver"] = risk_df.apply(
    dominant_driver,
    axis=1
)


# ============================================================
# AIR QUALITY — DEFRA BIRMINGHAM LADYWOOD 2025
# ============================================================

air_df = None
air_error = None


try:

    response = requests.get(
        AIR_2025_URL,
        timeout=60
    )

    response.raise_for_status()

    air_df = pd.read_csv(
        BytesIO(response.content)
    )

except Exception as e:

    air_error = str(e)


# ============================================================
# AIR COLUMN DETECTION
# ============================================================

if air_df is not None:

    air_df.columns = [
        str(column).strip()
        for column in air_df.columns
    ]

    date_column = None
    time_column = None

    for column in air_df.columns:

        lower = column.lower()

        if "date" in lower and "time" in lower:
            date_column = column
            break

    if date_column is None:

        for column in air_df.columns:

            if "date" in column.lower():

                date_column = column
                break

    for column in air_df.columns:

        lower = column.lower()

        if "time" in lower:

            time_column = column

            if column != date_column:
                break


    # Combine separate date and time columns if required.

    if date_column and time_column:

        air_df["DateTime"] = pd.to_datetime(
            air_df[date_column].astype(str)
            + " "
            + air_df[time_column].astype(str),
            errors="coerce"
        )

    elif date_column:

        air_df["DateTime"] = pd.to_datetime(
            air_df[date_column],
            errors="coerce"
        )

    else:

        air_df["DateTime"] = pd.NaT


    # Find NO2 and PM2.5 columns.

    no2_column = None
    pm25_column = None

    for column in air_df.columns:

        lower = column.lower()

        if (
            no2_column is None
            and "no2" in lower
        ):
            no2_column = column

        if (
            pm25_column is None
            and "pm2.5" in lower
        ):
            pm25_column = column


    if no2_column:

        air_df["NO2"] = pd.to_numeric(
            air_df[no2_column],
            errors="coerce"
        )

    else:

        air_df["NO2"] = pd.NA


    if pm25_column:

        air_df["PM2.5"] = pd.to_numeric(
            air_df[pm25_column],
            errors="coerce"
        )

    else:

        air_df["PM2.5"] = pd.NA


    air_df = air_df[
        air_df["DateTime"].notna()
    ].copy()


    if not air_df.empty:

        air_df["Month"] = (
            air_df["DateTime"]
            .dt.to_period("M")
            .astype(str)
        )


# ============================================================
# MAP
# ============================================================

center = ladywood_geometry.centroid

center_wgs = transform(
    to_wgs84,
    center
)


m = folium.Map(
    location=[
        center_wgs.y,
        center_wgs.x
    ],
    zoom_start=14,
    control_scale=True
)


# ============================================================
# MAP KEY
# ============================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 12px;
border: 1px solid grey;
font-size: 13px;
">
<b>Environmental Screening</b><br>
<span style="color:green;">●</span> LOW<br>
<span style="color:orange;">●</span> MODERATE<br>
<span style="color:red;">●</span> HIGH
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)


# ============================================================
# LADYWOOD BOUNDARY
# ============================================================

ladywood_wgs = transform(
    to_wgs84,
    ladywood_geometry
)

folium.GeoJson(
    mapping(ladywood_wgs),
    name="Official Ladywood Boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 3,
        "fill": False
    },
    tooltip="Ladywood Ward"
).add_to(m)


# ============================================================
# LSOA RISK LAYER
# ============================================================

for _, row in risk_df.iterrows():

    geom_wgs = transform(
        to_wgs84,
        row["geometry"]
    )

    risk = row["Risk_Class"]

    if risk == "HIGH":
        colour = "red"

    elif risk == "MODERATE":
        colour = "orange"

    else:
        colour = "green"

    popup_text = (
        f"<b>{row['LSOA21NM']}</b><br>"
        f"LSOA: {row['LSOA21CD']}<br>"
        f"Risk: {risk}<br>"
        f"Environmental Index: "
        f"{row['Environmental_Index']:.2f}<br>"
        f"Flood exposure: "
        f"{row['Combined_Flood_Exposure_%']:.1f}%<br>"
        f"Brownfield exposure: "
        f"{row['Brownfield_Exposure_%']:.1f}%<br>"
        f"Dominant driver: "
        f"{row['Dominant_Driver']}"
    )

    folium.GeoJson(
        mapping(geom_wgs),
        name="Official LSOA Areas",
        style_function=lambda feature,
        colour=colour: {
            "color": colour,
            "weight": 2,
            "fillColor": colour,
            "fillOpacity": 0.25
        },
        popup=folium.Popup(
            popup_text,
            max_width=350
        )
    ).add_to(m)


# ============================================================
# FLOOD ZONE 3
# ============================================================

flood_group = folium.FeatureGroup(
    name="Flood Zone 3",
    show=False
)

for item in flood3_features:

    geom_wgs = transform(
        to_wgs84,
        item["geometry"]
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "blue",
            "weight": 1,
            "fillColor": "blue",
            "fillOpacity": 0.20
        },
        tooltip="Flood Zone 3"
    ).add_to(flood_group)

flood_group.add_to(m)


# ============================================================
# SURFACE FLOODING
# ============================================================

surface_group = folium.FeatureGroup(
    name="Surface Flooding",
    show=False
)

for item in surface_flood_features:

    geom_wgs = transform(
        to_wgs84,
        item["geometry"]
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "purple",
            "weight": 1,
            "fillColor": "purple",
            "fillOpacity": 0.20
        },
        tooltip="Surface Flooding"
    ).add_to(surface_group)

surface_group.add_to(m)


# ============================================================
# BROWNFIELD REGISTER
# ============================================================

brownfield_group = folium.FeatureGroup(
    name="Brownfield Register 2025",
    show=False
)

for item in brownfield_features:

    geom_wgs = transform(
        to_wgs84,
        item["geometry"]
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "brown",
            "weight": 2,
            "fillColor": "brown",
            "fillOpacity": 0.25
        },
        tooltip="Birmingham Brownfield Register 2025"
    ).add_to(brownfield_group)

brownfield_group.add_to(m)


# ============================================================
# CLASSIFIED ROADS
# ============================================================

road_group = folium.FeatureGroup(
    name="Classified Roads",
    show=False
)

for item in road_features:

    geom = item["geometry"]

    if geom is None:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    attributes = item["attributes"]

    road_class = (
        attributes.get("BRUM_CLASS")
        or "Classified road"
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "black",
            "weight": 2
        },
        tooltip=str(road_class)
    ).add_to(road_group)

road_group.add_to(m)


# ============================================================
# RISK MARKERS
# ============================================================

for _, row in risk_df.iterrows():

    point = row["geometry"].centroid

    point_wgs = transform(
        to_wgs84,
        point
    )

    risk = row["Risk_Class"]

    if risk == "HIGH":
        icon_colour = "red"

    elif risk == "MODERATE":
        icon_colour = "orange"

    else:
        icon_colour = "green"

    folium.Marker(
        location=[
            point_wgs.y,
            point_wgs.x
        ],
        tooltip=(
            f"{row['LSOA21NM']} — "
            f"{risk}"
        ),
        popup=(
            f"<b>{row['LSOA21NM']}</b><br>"
            f"Risk: {risk}<br>"
            f"Environmental Index: "
            f"{row['Environmental_Index']:.2f}<br>"
            f"Dominant driver: "
            f"{row['Dominant_Driver']}"
        ),
        icon=folium.Icon(
            color=icon_colour,
            icon="info-sign"
        )
    ).add_to(m)


folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY MAP
# ============================================================

st.components.v1.html(
    m.get_root().render(),
    height=650,
    scrolling=False
)


# ============================================================
# ENVIRONMENTAL SCREENING
# ============================================================

st.subheader(
    "Environmental Screening Classification"
)

display_columns = [
    "LSOA21CD",
    "LSOA21NM",
    "CRVA_Average_Risk",
    "Combined_Flood_Exposure_%",
    "Brownfield_Exposure_%",
    "Environmental_Index",
    "Risk_Class",
    "Dominant_Driver"
]

st.dataframe(
    risk_df[
        display_columns
    ].sort_values(
        "Environmental_Index",
        ascending=False
    ),
    use_container_width=True
)


# ============================================================
# PRIORITY AREAS
# ============================================================

st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

priority = risk_df[
    risk_df["Risk_Class"] == "HIGH"
].sort_values(
    "Environmental_Index",
    ascending=False
)

if priority.empty:

    st.info(
        "No Ladywood LSOA reached the HIGH screening category "
        "using the stated project screening calculation."
    )

else:

    st.dataframe(
        priority[
            [
                "LSOA21CD",
                "LSOA21NM",
                "Environmental_Index",
                "Risk_Class",
                "Dominant_Driver"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# ENVIRONMENTAL INDEX CHART
# ============================================================

st.subheader(
    "Environmental Risk by Ladywood Area"
)

chart_data = (
    risk_df[
        [
            "LSOA21NM",
            "Environmental_Index"
        ]
    ]
    .set_index("LSOA21NM")
)

st.bar_chart(
    chart_data
)


# ============================================================
# FLOOD EXPOSURE
# ============================================================

st.subheader(
    "Flood Exposure by Ladywood Area"
)

flood_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Combined_Flood_Exposure_%"
        ]
    ]
    .set_index("LSOA21NM")
)

st.bar_chart(
    flood_chart
)


# ============================================================
# BROWNFIELD EXPOSURE
# ============================================================

st.subheader(
    "Brownfield Exposure by Ladywood Area"
)

brownfield_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Brownfield_Exposure_%"
        ]
    ]
    .set_index("LSOA21NM")
)

st.bar_chart(
    brownfield_chart
)


# ============================================================
# DEFRA AIR QUALITY
# ============================================================

st.subheader(
    "Measured Air Quality — Birmingham Ladywood, 2025"
)

st.caption(
    "Air measurements are from the Birmingham Ladywood "
    "DEFRA monitoring station. They are displayed separately "
    "from LSOA spatial screening because the monitoring station "
    "does not provide one monitor per LSOA."
)


if air_df is None:

    st.warning(
        "The DEFRA 2025 air-quality file could not be loaded."
    )

    if air_error:
        st.caption(air_error)

elif air_df.empty:

    st.warning(
        "The DEFRA file loaded but contained no usable dated records."
    )

else:

    monthly_air = (
        air_df
        .groupby("Month")[
            ["NO2", "PM2.5"]
        ]
        .mean()
    )

    st.markdown("**Monthly NO₂ average**")

    st.line_chart(
        monthly_air["NO2"]
    )

    st.markdown("**Monthly PM2.5 average**")

    st.line_chart(
        monthly_air["PM2.5"]
    )


# ============================================================
# CLASSIFIED ROADS
# ============================================================

st.subheader(
    "Classified Roads Within Ladywood"
)

road_table = []

for item in road_features:

    attributes = item["attributes"]

    road_table.append(
        {
            "Road classification":
                attributes.get("BRUM_CLASS"),
            "Road":
                attributes.get("BRUM_ROAD_")
        }
    )

if road_table:

    roads_df = pd.DataFrame(
        road_table
    )

    st.dataframe(
        roads_df.drop_duplicates(),
        use_container_width=True
    )

else:

    st.info(
        "No classified-road features were returned "
        "inside the official Ladywood boundary."
    )


# ============================================================
# ENGINEERING SCREENING CALCULATION
# ============================================================

st.subheader(
    "Engineering Screening Calculation"
)

st.latex(
    r"E=\frac{C+F+B}{3}"
)

st.caption(
    "E = Environmental Screening Index; "
    "C = normalised CRVA vulnerability; "
    "F = normalised flood exposure; "
    "B = normalised brownfield exposure."
)

st.latex(
    r"I=\frac{x-x_{\min}}{x_{\max}-x_{\min}}"
)

st.caption(
    "The calculation is a transparent project screening method. "
    "It is not presented as an official Birmingham risk rating "
    "or statutory environmental assessment."
)


# ============================================================
# DOWNLOAD
# ============================================================

download_df = risk_df.drop(
    columns=["geometry"]
)

csv = download_df.to_csv(
    index=False
)

st.download_button(
    label="Download Ladywood Environmental Results",
    data=csv,
    file_name="Ladywood_Environmental_Screening.csv",
    mime="text/csv"
)


# ============================================================
# DATA SOURCES
# ============================================================

st.subheader(
    "Official Data Sources"
)

st.write(
    "Birmingham City Council — CRVA 2025"
)

st.write(
    "Birmingham City Council — Brownfield Register"
)

st.write(
    "Birmingham City Council — Road Classification"
)

st.write(
    "DEFRA UK-AIR — Birmingham Ladywood monitoring station"
)
