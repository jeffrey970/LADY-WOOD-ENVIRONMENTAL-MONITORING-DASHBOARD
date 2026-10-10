import streamlit as st
import pandas as pd
import requests
import folium
import re

from io import BytesIO, StringIO
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    Point
)
from shapely.ops import transform, unary_union
from pyproj import Transformer


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Dashboard")

st.caption(
    "Ladywood, Birmingham, UK Environmental Dashboard"
)


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
    "Internet_Highways_open/MapServer/26"
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
# GENERAL HELPERS
# ============================================================

def clean_number(value):
    """Convert common DEFRA numeric formats to numbers."""

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text == "":
        return None

    text = text.replace(",", "")

    # DEFRA may contain < or > qualifiers.
    text = text.replace("<", "")
    text = text.replace(">", "")

    try:
        return float(text)
    except Exception:
        return None


def normalise_column_name(name):
    """Make column names easier to search."""

    text = str(name).lower().strip()

    text = text.replace("₂", "2")
    text = text.replace("₅", "5")

    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text
    )

    return text


# ============================================================
# ARC GIS
# ============================================================

def arcgis_request(
    layer_url,
    where="1=1",
    geometry=None,
    result_offset=0
):

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "27700",
        "f": "json"
    }

    if geometry is not None:

        params["geometry"] = ",".join(
            str(round(v, 2))
            for v in geometry
        )

        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"

    if result_offset:
        params["resultOffset"] = result_offset
        params["resultRecordCount"] = 1000

    response = requests.get(
        layer_url + "/query",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            str(data["error"])
        )

    return data


# ============================================================
# ESRI GEOMETRY CONVERSION
# ============================================================

def esri_geometry_to_shape(geometry):

    if not geometry:
        return None

    # --------------------------------------------------------
    # POLYGONS
    # --------------------------------------------------------

    if "rings" in geometry:

        rings = geometry.get("rings", [])

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
                pass

        if not polygons:
            return None

        if len(polygons) == 1:
            return polygons[0]

        return unary_union(polygons)

    # --------------------------------------------------------
    # LINES
    # --------------------------------------------------------

    if "paths" in geometry:

        paths = geometry.get("paths", [])

        lines = []

        for path in paths:

            if len(path) < 2:
                continue

            try:

                line = LineString(path)

                if not line.is_empty:
                    lines.append(line)

            except Exception:
                pass

        if not lines:
            return None

        if len(lines) == 1:
            return lines[0]

        return MultiLineString(lines)

    # --------------------------------------------------------
    # POINTS
    # --------------------------------------------------------

    if "x" in geometry and "y" in geometry:

        return Point(
            geometry["x"],
            geometry["y"]
        )

    return None


# ============================================================
# GET ARC GIS FEATURES
# ============================================================

def get_features(
    layer_url,
    where="1=1",
    geometry=None
):

    first = arcgis_request(
        layer_url,
        where=where,
        geometry=geometry
    )

    features = first.get(
        "features",
        []
    )

    exceeded = first.get(
        "exceededTransferLimit",
        False
    )

    offset = len(features)

    while exceeded:

        next_data = arcgis_request(
            layer_url,
            where=where,
            geometry=geometry,
            result_offset=offset
        )

        next_features = next_data.get(
            "features",
            []
        )

        features.extend(
            next_features
        )

        if not next_data.get(
            "exceededTransferLimit",
            False
        ):
            break

        offset += len(next_features)

        if not next_features:
            break

    result = []

    for feature in features:

        attributes = feature.get(
            "attributes",
            {}
        ).copy()

        attributes["geometry"] = (
            esri_geometry_to_shape(
                feature.get("geometry")
            )
        )

        result.append(
            attributes
        )

    return result


# ============================================================
# OFFICIAL LADYWOOD BOUNDARY
# ============================================================

ladywood_rows = []

try:

    ladywood_rows = get_features(
        WARD_LAYER,
        where="WARDNME = 'Ladywood'"
    )

except Exception:
    ladywood_rows = []


# Fallback: download wards and find Ladywood locally.
if not ladywood_rows:

    try:

        all_wards = get_features(
            WARD_LAYER
        )

        for row in all_wards:

            ward_name = str(
                row.get(
                    "WARDNME",
                    ""
                )
            ).strip().lower()

            if ward_name == "ladywood":

                ladywood_rows = [
                    row
                ]

                break

    except Exception:
        ladywood_rows = []


if not ladywood_rows:

    st.error(
        "The official Ladywood ward boundary could not be retrieved."
    )

    st.stop()


ladywood_geometry = (
    ladywood_rows[0]
    .get("geometry")
)


if (
    ladywood_geometry is None
    or ladywood_geometry.is_empty
):

    st.error(
        "The official Ladywood ward was found, "
        "but its geometry was empty."
    )

    st.stop()


ladywood_geometry = (
    ladywood_geometry
    .buffer(0)
)


# ============================================================
# LADYWOOD EXTENT
# ============================================================

ladywood_bbox = (
    *ladywood_geometry.bounds,
)


# ============================================================
# OFFICIAL LSOAs
# ============================================================

try:

    lsoa_rows = get_features(
        LSOA_LAYER,
        geometry=ladywood_bbox
    )

except Exception as error:

    st.error(
        "The official LSOA dataset could not be loaded."
    )

    st.caption(str(error))

    st.stop()


lsoa_records = []


for row in lsoa_rows:

    geom = row.get(
        "geometry"
    )

    if geom is None:
        continue

    if geom.is_empty:
        continue

    # Only keep the portion inside Ladywood.
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
            "LSOA21CD": row.get(
                "LSOA21CD"
            ),
            "LSOA21NM": row.get(
                "LSOA21NM"
            ),
            "MEAN": row.get(
                "MEAN"
            ),
            "MIN": row.get(
                "MIN"
            ),
            "MAX": row.get(
                "MAX"
            ),
            "MEDIAN": row.get(
                "MEDIAN"
            ),
            "AVERAGE_RISK": row.get(
                "AVERAGE_RISK"
            ),
            "MINIMUM_RISK": row.get(
                "MINIMUM_RISK"
            ),
            "MAXIMUM_RISK": row.get(
                "MAXIMUM_RISK"
            ),
            "geometry": clipped,
            "Area_m2": area
        }
    )


lsoa_df = pd.DataFrame(
    lsoa_records
)


if lsoa_df.empty:

    st.error(
        "No official LSOA areas intersecting Ladywood were found."
    )

    st.stop()


# Convert official MEAN to numeric.
lsoa_df["CRVA_MEAN"] = pd.to_numeric(
    lsoa_df["MEAN"],
    errors="coerce"
)


# ============================================================
# SPATIAL DATA LOADER
# ============================================================

def load_spatial_layer(
    layer_url
):

    try:

        rows = get_features(
            layer_url,
            geometry=ladywood_bbox
        )

    except Exception:

        return []

    features = []

    for row in rows:

        geom = row.get(
            "geometry"
        )

        if geom is None:
            continue

        if geom.is_empty:
            continue

        # Only retain features actually intersecting Ladywood.
        if not geom.intersects(
            ladywood_geometry
        ):
            continue

        clipped = geom.intersection(
            ladywood_geometry
        )

        if clipped.is_empty:
            continue

        features.append(
            {
                "attributes": row,
                "geometry": clipped
            }
        )

    return features


# ============================================================
# LOAD ENVIRONMENTAL DATA
# ============================================================

flood3_features = load_spatial_layer(
    FLOOD_ZONE_3_LAYER
)

surface_features = load_spatial_layer(
    SURFACE_FLOOD_LAYER
)

brownfield_features = load_spatial_layer(
    BROWNFIELD_LAYER
)

road_features = load_spatial_layer(
    ROAD_LAYER
)


# ============================================================
# EXPOSURE CALCULATION
# ============================================================

def exposure_percent(
    lsoa_geometry,
    features
):

    if not features:
        return 0.0

    intersections = []

    for feature in features:

        geom = feature.get(
            "geometry"
        )

        if geom is None:
            continue

        if not geom.intersects(
            lsoa_geometry
        ):
            continue

        intersection = geom.intersection(
            lsoa_geometry
        )

        if not intersection.is_empty:

            intersections.append(
                intersection
            )

    if not intersections:
        return 0.0

    combined = unary_union(
        intersections
    )

    exposed_area = combined.area

    total_area = (
        lsoa_geometry.area
    )

    if total_area <= 0:
        return 0.0

    return (
        exposed_area /
        total_area
    ) * 100


# ============================================================
# BUILD ENVIRONMENTAL RESULTS
# ============================================================

results = []


for _, row in lsoa_df.iterrows():

    geom = row["geometry"]

    flood_zone_3 = exposure_percent(
        geom,
        flood3_features
    )

    surface_flood = exposure_percent(
        geom,
        surface_features
    )

    brownfield = exposure_percent(
        geom,
        brownfield_features
    )

    # Avoid double-counting overlapping flood layers.
    combined_flood = max(
        flood_zone_3,
        surface_flood
    )

    results.append(
        {
            "LSOA21CD":
                row["LSOA21CD"],

            "LSOA21NM":
                row["LSOA21NM"],

            "Area_m2":
                row["Area_m2"],

            "CRVA_MEAN":
                row["CRVA_MEAN"],

            "CRVA_AVERAGE_RISK":
                row["AVERAGE_RISK"],

            "Flood_Zone_3_%":
                flood_zone_3,

            "Surface_Flood_%":
                surface_flood,

            "Combined_Flood_Exposure_%":
                combined_flood,

            "Brownfield_Exposure_%":
                brownfield,

            "geometry":
                geom
        }
    )


risk_df = pd.DataFrame(
    results
)


# ============================================================
# NORMALISATION
# ============================================================

def normalise(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = numeric.min()
    maximum = numeric.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or maximum == minimum
    ):

        return pd.Series(
            [0.0] * len(series),
            index=series.index
        )

    return (
        numeric - minimum
    ) / (
        maximum - minimum
    )


risk_df["Climate_Index"] = normalise(
    risk_df["CRVA_MEAN"]
)

risk_df["Flood_Index"] = normalise(
    risk_df[
        "Combined_Flood_Exposure_%"
    ]
)

risk_df["Brownfield_Index"] = normalise(
    risk_df[
        "Brownfield_Exposure_%"
    ]
)


# ============================================================
# ENVIRONMENTAL SCREENING INDEX
# ============================================================

risk_df["Environmental_Index"] = (
    risk_df["Climate_Index"] +
    risk_df["Flood_Index"] +
    risk_df["Brownfield_Index"]
) / 3


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(value):

    if pd.isna(value):
        return "NOT AVAILABLE"

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


risk_df["Risk_Class"] = (
    risk_df[
        "Environmental_Index"
    ]
    .apply(classify_risk)
)


# ============================================================
# DOMINANT DRIVER
# ============================================================

def dominant_driver(row):

    values = {
        "Climate vulnerability":
            row["Climate_Index"],

        "Flood exposure":
            row["Flood_Index"],

        "Brownfield exposure":
            row["Brownfield_Index"]
    }

    valid = {
        key: value
        for key, value in values.items()
        if pd.notna(value)
    }

    if not valid:
        return "Not available"

    return max(
        valid,
        key=valid.get
    )


risk_df[
    "Dominant_Environmental_Driver"
] = risk_df.apply(
    dominant_driver,
    axis=1
)


# ============================================================
# ROAD PROXIMITY
# ============================================================

def nearest_road_info(
    lsoa_geometry,
    roads
):

    if not roads:
        return (
            None,
            "No classified road data"
        )

    nearest_distance = None
    nearest_class = None

    for road in roads:

        geom = road.get(
            "geometry"
        )

        if geom is None:
            continue

        distance = (
            lsoa_geometry
            .distance(geom)
        )

        attributes = road.get(
            "attributes",
            {}
        )

        road_class = attributes.get(
            "BRUM_CLASS"
        )

        if (
            nearest_distance is None
            or distance < nearest_distance
        ):

            nearest_distance = distance
            nearest_class = road_class

    return (
        nearest_distance,
        nearest_class
    )


road_distances = []


for _, row in risk_df.iterrows():

    distance, road_class = (
        nearest_road_info(
            row["geometry"],
            road_features
        )
    )

    road_distances.append(
        {
            "Distance_m": distance,
            "Road_Class": road_class
        }
    )


risk_df[
    "Nearest_Classified_Road_m"
] = [
    item["Distance_m"]
    for item in road_distances
]

risk_df[
    "Nearest_Road_Class"
] = [
    item["Road_Class"]
    for item in road_distances
]


# ============================================================
# AIR QUALITY — DEFRA BIRMINGHAM LADYWOOD 2025
# ============================================================

air_df = None
air_error = None


try:

    response = requests.get(
        AIR_2025_URL,
        timeout=90
    )

    response.raise_for_status()

    raw_text = response.content.decode(
        "utf-8-sig",
        errors="replace"
    )

    # --------------------------------------------------------
    # Find the real CSV header.
    # DEFRA files can contain information before the table.
    # --------------------------------------------------------

    lines = raw_text.splitlines()

    header_line = None

    for i, line in enumerate(lines):

        lower = line.lower()

        if (
            "date" in lower
            and "time" in lower
        ):

            header_line = i
            break

    if header_line is None:
        header_line = 0

    csv_text = "\n".join(
        lines[header_line:]
    )

    air_df = pd.read_csv(
        StringIO(csv_text),
        sep=None,
        engine="python"
    )


except Exception as error:

    air_error = str(error)


# ============================================================
# PROCESS DEFRA DATA
# ============================================================

if air_df is not None:

    air_df.columns = [
        str(column).strip()
        for column in air_df.columns
    ]

    normalised_columns = {
        column:
        normalise_column_name(column)
        for column in air_df.columns
    }


    # --------------------------------------------------------
    # DATE/TIME
    # --------------------------------------------------------

    date_column = None
    time_column = None

    for column, normalised in (
        normalised_columns.items()
    ):

        if normalised == "date":
            date_column = column

        if normalised == "time":
            time_column = column


    if date_column and time_column:

        air_df["DateTime"] = pd.to_datetime(
            air_df[date_column].astype(str)
            + " "
            + air_df[time_column].astype(str),
            errors="coerce",
            dayfirst=True
        )

    elif date_column:

        air_df["DateTime"] = pd.to_datetime(
            air_df[date_column],
            errors="coerce",
            dayfirst=True
        )

    else:

        # Last attempt: find a combined date/time column.
        combined_column = None

        for column in air_df.columns:

            lower = column.lower()

            if (
                "date" in lower
                and "time" in lower
            ):

                combined_column = column
                break

        if combined_column:

            air_df["DateTime"] = pd.to_datetime(
                air_df[combined_column],
                errors="coerce",
                dayfirst=True
            )

        else:

            air_df["DateTime"] = pd.NaT


    # --------------------------------------------------------
    # POLLUTANT COLUMNS
    # --------------------------------------------------------

    no2_column = None
    pm25_column = None


    for column, normalised in (
        normalised_columns.items()
    ):

        # NO2 but not NOx.
        if (
            no2_column is None
            and (
                normalised == "no2"
                or normalised.startswith("no2")
            )
            and "status" not in normalised
        ):

            no2_column = column


        # PM2.5
        if (
            pm25_column is None
            and (
                "pm25" in normalised
                or "pm2p5" in normalised
            )
            and "status" not in normalised
        ):

            pm25_column = column


    if no2_column:

        air_df["NO2"] = (
            air_df[no2_column]
            .apply(clean_number)
        )

    else:

        air_df["NO2"] = pd.NA


    if pm25_column:

        air_df["PM2.5"] = (
            air_df[pm25_column]
            .apply(clean_number)
        )

    else:

        air_df["PM2.5"] = pd.NA


    # --------------------------------------------------------
    # Remove records without a valid date.
    # --------------------------------------------------------

    air_df = air_df[
        air_df["DateTime"].notna()
    ].copy()


    if not air_df.empty:

        air_df["Month"] = (
            air_df["DateTime"]
            .dt.to_period("M")
            .astype(str)
        )

        air_df["Day"] = (
            air_df["DateTime"]
            .dt.date
        )


# ============================================================
# MAP
# ============================================================

ladywood_center = (
    ladywood_geometry.centroid
)

center_wgs = transform(
    to_wgs84,
    ladywood_center
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
# MAP LEGEND
# ============================================================

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background-color: white;
padding: 12px;
border: 1px solid #777;
font-size: 13px;
box-shadow: 0 1px 5px rgba(0,0,0,0.3);
">
<b>Environmental Screening</b><br>
<span style="color:green;">●</span> Low<br>
<span style="color:orange;">●</span> Moderate<br>
<span style="color:red;">●</span> High<br>
<span style="color:grey;">●</span> Not available
</div>
"""

m.get_root().html.add_child(
    folium.Element(
        legend_html
    )
)


# ============================================================
# OFFICIAL LADYWOOD BOUNDARY
# ============================================================

ladywood_wgs = transform(
    to_wgs84,
    ladywood_geometry
)

folium.GeoJson(
    {
        "type": "Feature",
        "geometry": {
            "type": (
                "MultiPolygon"
                if isinstance(
                    ladywood_wgs,
                    MultiPolygon
                )
                else "Polygon"
            ),
            "coordinates":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    ladywood_wgs
                )["coordinates"]
        },
        "properties": {}
    },
    name="Ladywood Boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 3,
        "fillOpacity": 0
    },
    tooltip="Official Ladywood Ward Boundary"
).add_to(m)


# ============================================================
# LSOA RISK MAP
# ============================================================

lsoa_group = folium.FeatureGroup(
    name="Ladywood LSOA Risk",
    show=True
)


def risk_colour(value):

    if value == "HIGH":
        return "red"

    if value == "MODERATE":
        return "orange"

    if value == "LOW":
        return "green"

    return "gray"


for _, row in risk_df.iterrows():

    geom_wgs = transform(
        to_wgs84,
        row["geometry"]
    )

    risk = row["Risk_Class"]

    colour = risk_colour(
        risk
    )

    popup = f"""
    <b>{row['LSOA21NM']}</b><br>
    LSOA: {row['LSOA21CD']}<br>
    Screening class: {risk}<br>
    Environmental index:
    {row['Environmental_Index']:.2f}<br>
    Climate vulnerability:
    {row['Climate_Index']:.2f}<br>
    Flood exposure:
    {row['Combined_Flood_Exposure_%']:.1f}%<br>
    Brownfield exposure:
    {row['Brownfield_Exposure_%']:.1f}%<br>
    Dominant driver:
    {row['Dominant_Environmental_Driver']}<br>
    Nearest classified road:
    {
        f"{row['Nearest_Classified_Road_m']:.0f} m"
        if pd.notna(
            row["Nearest_Classified_Road_m"]
        )
        else "Not available"
    }
    """

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    geom_wgs
                ),
            "properties": {}
        },
        style_function=lambda feature,
        colour=colour: {
            "color": colour,
            "weight": 2,
            "fillColor": colour,
            "fillOpacity": 0.25
        },
        popup=folium.Popup(
            popup,
            max_width=400
        )
    ).add_to(
        lsoa_group
    )


lsoa_group.add_to(m)


# ============================================================
# FLOOD ZONE 3
# ============================================================

flood_group = folium.FeatureGroup(
    name="Flood Zone 3",
    show=False
)

for feature in flood3_features:

    geom_wgs = transform(
        to_wgs84,
        feature["geometry"]
    )

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    geom_wgs
                ),
            "properties": {}
        },
        style_function=lambda feature: {
            "color": "blue",
            "weight": 1,
            "fillColor": "blue",
            "fillOpacity": 0.25
        },
        tooltip="Flood Zone 3"
    ).add_to(
        flood_group
    )

flood_group.add_to(m)


# ============================================================
# SURFACE FLOODING
# ============================================================

surface_group = folium.FeatureGroup(
    name="Surface Flooding",
    show=False
)

for feature in surface_features:

    geom_wgs = transform(
        to_wgs84,
        feature["geometry"]
    )

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    geom_wgs
                ),
            "properties": {}
        },
        style_function=lambda feature: {
            "color": "purple",
            "weight": 1,
            "fillColor": "purple",
            "fillOpacity": 0.20
        },
        tooltip="Surface Flooding"
    ).add_to(
        surface_group
    )

surface_group.add_to(m)


# ============================================================
# BROWNFIELD
# ============================================================

brownfield_group = folium.FeatureGroup(
    name="Brownfield Register 2025",
    show=False
)

for feature in brownfield_features:

    geom_wgs = transform(
        to_wgs84,
        feature["geometry"]
    )

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    geom_wgs
                ),
            "properties": {}
        },
        style_function=lambda feature: {
            "color": "brown",
            "weight": 2,
            "fillColor": "brown",
            "fillOpacity": 0.25
        },
        tooltip="Brownfield Register 2025"
    ).add_to(
        brownfield_group
    )

brownfield_group.add_to(m)


# ============================================================
# CLASSIFIED ROADS
# ============================================================

road_group = folium.FeatureGroup(
    name="Classified Roads",
    show=False
)


def road_colour(road_class):

    if road_class == "A Road":
        return "red"

    if road_class == "B Road":
        return "blue"

    return "gold"


for feature in road_features:

    geom = feature["geometry"]

    if geom is None:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    road_class = (
        feature["attributes"]
        .get(
            "BRUM_CLASS",
            "Classified Road"
        )
    )

    road_name = (
        feature["attributes"]
        .get(
            "BRUM_ROAD_",
            ""
        )
    )

    folium.GeoJson(
        {
            "type": "Feature",
            "geometry":
                __import__(
                    "shapely"
                ).geometry.mapping(
                    geom_wgs
                ),
            "properties": {}
        },
        style_function=lambda feature,
        road_class=road_class: {
            "color":
                road_colour(
                    road_class
                ),
            "weight": 3
        },
        tooltip=(
            f"{road_class}"
            + (
                f" — {road_name}"
                if road_name
                else ""
            )
        )
    ).add_to(
        road_group
    )


road_group.add_to(m)


# ============================================================
# RISK DOTS
# ============================================================

marker_group = folium.FeatureGroup(
    name="Risk Markers",
    show=True
)


for _, row in risk_df.iterrows():

    point = (
        row["geometry"]
        .centroid
    )

    point_wgs = transform(
        to_wgs84,
        point
    )

    colour = risk_colour(
        row["Risk_Class"]
    )

    folium.CircleMarker(
        location=[
            point_wgs.y,
            point_wgs.x
        ],
        radius=8,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.9,
        tooltip=(
            f"{row['LSOA21NM']} — "
            f"{row['Risk_Class']}"
        ),
        popup=folium.Popup(
            f"""
            <b>{row['LSOA21NM']}</b><br>
            Environmental index:
            {row['Environmental_Index']:.2f}<br>
            Risk:
            {row['Risk_Class']}<br>
            Main driver:
            {row['Dominant_Environmental_Driver']}
            """,
            max_width=350
        )
    ).add_to(
        marker_group
    )


marker_group.add_to(m)


# ============================================================
# LAYER CONTROL
# ============================================================

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# SHOW MAP
# ============================================================

st.subheader(
    "Ladywood Environmental Map"
)

st.caption(
    "Use the layer control in the top-right of the map to "
    "turn environmental layers on and off. The coloured "
    "markers identify the screening result for each official "
    "Ladywood LSOA."
)

st.components.v1.html(
    m.get_root().render(),
    height=680,
    scrolling=False
)


# ============================================================
# ENVIRONMENTAL SCREENING TABLE
# ============================================================

st.subheader(
    "Environmental Screening Classification"
)

st.caption(
    "This table compares the official Ladywood LSOA areas "
    "using climate/environmental vulnerability, flood exposure "
    "and brownfield exposure. The indicators are normalised "
    "before being combined into a transparent screening index. "
    "Missing source data are not converted into artificial zeros."
)


table_columns = [
    "LSOA21CD",
    "LSOA21NM",
    "CRVA_MEAN",
    "CRVA_AVERAGE_RISK",
    "Combined_Flood_Exposure_%",
    "Brownfield_Exposure_%",
    "Environmental_Index",
    "Risk_Class",
    "Dominant_Environmental_Driver"
]

display_df = risk_df[
    table_columns
].copy()

display_df = display_df.sort_values(
    "Environmental_Index",
    ascending=False
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PRIORITY AREAS
# ============================================================

st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

st.caption(
    "Priority is based on the environmental screening index. "
    "This is an engineering screening calculation for the "
    "project, not an official Birmingham statutory risk rating."
)


st.markdown(
    """
**Screening calculation**

\[
E = \\frac{C + F + B}{3}
\]

where:

- **C** = normalised climate/environmental vulnerability
- **F** = normalised flood exposure
- **B** = normalised brownfield exposure

Risk classification:

- **HIGH:** E ≥ 0.67
- **MODERATE:** 0.34 ≤ E < 0.67
- **LOW:** E < 0.34
"""
)


priority_df = risk_df[
    risk_df["Risk_Class"] == "HIGH"
].sort_values(
    "Environmental_Index",
    ascending=False
)


if priority_df.empty:

    st.info(
        "No Ladywood LSOA reached the HIGH screening "
        "category using this calculation."
    )

    moderate_df = risk_df[
        risk_df["Risk_Class"] == "MODERATE"
    ].sort_values(
        "Environmental_Index",
        ascending=False
    )

    if not moderate_df.empty:

        st.write(
            "**Highest MODERATE screening areas:**"
        )

        st.dataframe(
            moderate_df[
                [
                    "LSOA21CD",
                    "LSOA21NM",
                    "Environmental_Index",
                    "Dominant_Environmental_Driver"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

else:

    st.dataframe(
        priority_df[
            [
                "LSOA21CD",
                "LSOA21NM",
                "Environmental_Index",
                "Dominant_Environmental_Driver"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CHART 1 — ENVIRONMENTAL RISK
# ============================================================

st.subheader(
    "Environmental Risk by Ladywood Area"
)

st.caption(
    "Higher values indicate higher combined screening scores "
    "from climate/environmental vulnerability, flood exposure "
    "and brownfield exposure."
)

environment_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Environmental_Index"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    environment_chart
)


# ============================================================
# CHART 2 — CLIMATE
# ============================================================

st.subheader(
    "Climate / Environmental Vulnerability by Ladywood Area"
)

st.caption(
    "This uses the official numeric MEAN value provided in "
    "Birmingham's 2025 LSOA environmental-risk dataset and "
    "normalises it for comparison between Ladywood areas."
)

climate_chart = (
    risk_df[
        [
            "LSOA21NM",
            "CRVA_MEAN"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    climate_chart
)


# ============================================================
# CHART 3 — FLOOD
# ============================================================

st.subheader(
    "Flood Exposure by Ladywood Area"
)

st.caption(
    "Percentage of each official Ladywood LSOA intersecting "
    "the combined flood screening extent. Flood Zone 3 and "
    "surface flooding are kept as separate map layers."
)

flood_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Combined_Flood_Exposure_%"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    flood_chart
)


# ============================================================
# CHART 4 — FLOOD DETAIL
# ============================================================

st.subheader(
    "Flood Exposure Components")

flood_detail = (
    risk_df[
        [
            "LSOA21NM",
            "Flood_Zone_3_%",
            "Surface_Flood_%"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    flood_detail
)


# ============================================================
# CHART 5 — BROWNFIELD
# ============================================================

st.subheader(
    "Brownfield Exposure by Ladywood Area"
)

st.caption(
    "Percentage of each official Ladywood LSOA intersecting "
    "the Birmingham Brownfield Register 2025."
)

brownfield_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Brownfield_Exposure_%"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    brownfield_chart
)


# ============================================================
# CHART 6 — DRIVER COMPARISON
# ============================================================

st.subheader(
    "Environmental Drivers by Ladywood Area"
)

st.caption(
    "Normalised indicators allow the different environmental "
    "factors to be compared on the same 0–1 scale."
)

driver_chart = (
    risk_df[
        [
            "LSOA21NM",
            "Climate_Index",
            "Flood_Index",
            "Brownfield_Index"
        ]
    ]
    .set_index(
        "LSOA21NM"
    )
)

st.bar_chart(
    driver_chart
)


# ============================================================
# DEFRA AIR QUALITY
# ============================================================

st.subheader(
    "Measured Air Quality — Birmingham Ladywood, 2025"
)

st.caption(
    "Measured at the official Birmingham Ladywood DEFRA "
    "monitoring site. These measurements describe the Ladywood "
    "monitoring location and are not treated as if each LSOA "
    "has its own air-quality monitor."
)


if air_df is None:

    st.error(
        "The DEFRA Birmingham Ladywood 2025 file could not "
        "be loaded."
    )

    if air_error:
        st.caption(
            "Technical detail: "
            + air_error
        )


elif air_df.empty:

    st.warning(
        "The DEFRA file loaded, but no valid dated records "
        "were available."
    )


else:

    # --------------------------------------------------------
    # AIR DATA SUMMARY
    # --------------------------------------------------------

    no2_count = (
        air_df["NO2"]
        .notna()
        .sum()
    )

    pm25_count = (
        air_df["PM2.5"]
        .notna()
        .sum()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "NO₂ valid hourly records",
            f"{no2_count:,}"
        )

    with col2:

        st.metric(
            "PM₂.₅ valid hourly records",
            f"{pm25_count:,}"
        )

    with col3:

        st.metric(
            "Months available",
            air_df["Month"].nunique()
        )


    # --------------------------------------------------------
    # MONTHLY AVERAGES
    # --------------------------------------------------------

    monthly_air = (
        air_df
        .groupby("Month")[
            ["NO2", "PM2.5"]
        ]
        .mean()
    )


    st.markdown(
        "### Monthly NO₂ trend"
    )

    st.line_chart(
        monthly_air["NO2"]
    )


    st.markdown(
        "### Monthly PM₂.₅ trend"
    )

    st.line_chart(
        monthly_air["PM2.5"]
    )


    st.markdown(
        "### Monthly NO₂ and PM₂.₅"
    )

    st.line_chart(
        monthly_air[
            [
                "NO2",
                "PM2.5"
            ]
        ]
    )


    # --------------------------------------------------------
    # AIR MONTHLY TABLE
    # --------------------------------------------------------

    air_monthly_display = (
        monthly_air
        .rename(
            columns={
                "NO2":
                    "NO2 monthly mean",
                "PM2.5":
                    "PM2.5 monthly mean"
            }
        )
    )

    st.dataframe(
        air_monthly_display,
        use_container_width=True
    )


    # --------------------------------------------------------
    # PM4.5
    # --------------------------------------------------------

    st.markdown(
        "### PM₄.₅"
    )

    st.info(
        "PM₄.₅ is not substituted with PM₂.₅ or PM₁₀. "
        "The official Birmingham Ladywood DEFRA dataset used "
        "here provides PM₂.₅, but does not provide a PM₄.₅ "
        "measurement. Therefore no PM₄.₅ values are invented."
    )


# ============================================================
# CLASSIFIED ROADS
# ============================================================

st.subheader(
    "Classified Roads Within Ladywood"
)

st.caption(
    "Official Birmingham classified-road data are shown here "
    "as environmental context. Road proximity is reported "
    "separately rather than being given an arbitrary pollution "
    "multiplier."
)


road_records = []


for feature in road_features:

    attributes = feature.get(
        "attributes",
        {}
    )

    road_records.append(
        {
            "Road classification":
                attributes.get(
                    "BRUM_CLASS"
                ),

            "Road":
                attributes.get(
                    "BRUM_ROAD_"
                )
        }
    )


if road_records:

    roads_df = pd.DataFrame(
        road_records
    )

    roads_df = (
        roads_df
        .drop_duplicates()
        .sort_values(
            [
                "Road classification",
                "Road"
            ],
            na_position="last"
        )
    )

    st.dataframe(
        roads_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "No classified-road features were returned inside "
        "the official Ladywood boundary."
    )


# ============================================================
# ROAD PROXIMITY BY LSOA
# ============================================================

st.subheader(
    "Classified Road Proximity by Ladywood Area"
)

road_proximity_df = risk_df[
    [
        "LSOA21CD",
        "LSOA21NM",
        "Nearest_Classified_Road_m",
        "Nearest_Road_Class"
    ]
].copy()

road_proximity_df = (
    road_proximity_df
    .sort_values(
        "Nearest_Classified_Road_m",
        na_position="last"
    )
)

st.dataframe(
    road_proximity_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================

st.subheader(
    "Engineering Screening Interpretation"
)

st.caption(
    "The dashboard applies a source → pathway → receptor "
    "screening approach. Environmental indicators identify "
    "where further engineering investigation may be useful; "
    "they do not replace site investigation or statutory "
    "environmental assessment."
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

download_columns = [
    "LSOA21CD",
    "LSOA21NM",
    "Area_m2",
    "CRVA_MEAN",
    "CRVA_AVERAGE_RISK",
    "Climate_Index",
    "Flood_Zone_3_%",
    "Surface_Flood_%",
    "Combined_Flood_Exposure_%",
    "Flood_Index",
    "Brownfield_Exposure_%",
    "Brownfield_Index",
    "Environmental_Index",
    "Risk_Class",
    "Dominant_Environmental_Driver",
    "Nearest_Classified_Road_m",
    "Nearest_Road_Class"
]

download_df = risk_df[
    download_columns
].copy()


st.download_button(
    label="Download Ladywood Environmental Screening Results",
    data=download_df.to_csv(
        index=False
    ),
    file_name=(
        "Ladywood_Environmental_Screening.csv"
    ),
    mime="text/csv"
)


# ============================================================
# AIR DOWNLOAD
# ============================================================

if air_df is not None and not air_df.empty:

    monthly_download = (
        air_df
        .groupby("Month")[
            [
                "NO2",
                "PM2.5"
            ]
        ]
        .mean()
        .reset_index()
    )

    st.download_button(
        label="Download Ladywood 2025 Air Monthly Results",
        data=monthly_download.to_csv(
            index=False
        ),
        file_name=(
            "Birmingham_Ladywood_2025_Air_Monthly.csv"
        ),
        mime="text/csv"
    )
