import streamlit as st
import pandas as pd
import requests
import folium
import re

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    LineString,
    MultiLineString,
    Point,
    mapping,
    shape,
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
st.caption("Ladywood, Birmingham, UK")


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Boundaries/MapServer/0"
)

LSOA_LAYER = CRVA + "/17"

# Flood Zone 3 - High Probability
FLOOD_ZONE_3_LAYER = CRVA + "/109"

# Surface-water flooding
SURFACE_FLOOD_LAYER = CRVA + "/1546"

BROWNFIELD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROAD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Highways_open/MapServer/26"
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
# SESSION AND GENERAL HELPERS
# ============================================================

session = requests.Session()
session.headers.update({
    "User-Agent": "LadywoodEnvironmentalDashboard/1.0"
})


def normalise_column_name(name):
    text = str(name).lower().strip()
    text = text.replace("₂", "2")
    text = text.replace("₅", "5")
    return re.sub(r"[^a-z0-9]+", "", text)


# ============================================================
# ARCGIS REQUEST
# ============================================================

def arcgis_request(url, params=None):
    request_params = {"f": "json"}

    if params:
        request_params.update(params)

    response = session.get(
        url,
        params=request_params,
        timeout=90
    )
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            data["error"].get("message", str(data["error"]))
        )

    return data


# ============================================================
# ESRI GEOMETRY CONVERSION
# ============================================================

def esri_geometry_to_shape(geometry):
    if not geometry:
        return None

    # Polygons
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
                continue

        if not polygons:
            return None

        if len(polygons) == 1:
            return polygons[0]

        return unary_union(polygons)

    # Lines
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
                continue

        if not lines:
            return None

        if len(lines) == 1:
            return lines[0]

        return MultiLineString(lines)

    # Points
    if "x" in geometry and "y" in geometry:
        return Point(
            geometry["x"],
            geometry["y"]
        )

    return None


# ============================================================
# GET ARCGIS FEATURES
# ============================================================

def get_features(layer_url, geometry=None, where="1=1"):
    """
    Retrieve ArcGIS features using object IDs in batches.
    Returned geometries use EPSG:27700.
    """

    id_params = {
        "where": where,
        "returnIdsOnly": "true",
        "returnGeometry": "false",
    }

    if geometry is not None:
        xmin, ymin, xmax, ymax = geometry

        id_params.update({
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "27700",
            "spatialRel": "esriSpatialRelIntersects",
        })

    id_result = arcgis_request(
        layer_url + "/query",
        id_params
    )

    object_ids = id_result.get("objectIds") or []

    if not object_ids:
        return []

    all_features = []
    batch_size = 200

    for start in range(0, len(object_ids), batch_size):
        batch = object_ids[start:start + batch_size]

        result = arcgis_request(
            layer_url + "/query",
            {
                "objectIds": ",".join(
                    str(object_id) for object_id in batch
                ),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "27700",
            }
        )

        all_features.extend(
            result.get("features", [])
        )

    records = []

    for feature in all_features:
        attributes = feature.get(
            "attributes", {}
        ).copy()

        attributes["geometry"] = esri_geometry_to_shape(
            feature.get("geometry")
        )

        records.append(attributes)

    return records


# ============================================================
# GEOMETRY CLEANING
# ============================================================

def clean_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None

    try:
        if not geometry.is_valid:
            geometry = geometry.buffer(0)

        if geometry.is_empty:
            return None

        return geometry

    except Exception:
        return None


# ============================================================
# OFFICIAL LADYWOOD BOUNDARY
# ============================================================

def get_ladywood_boundary():
    """
    Retrieve the official ward layer and identify Ladywood using
    the WARDNME field. If the field name differs, check the returned
    attributes instead of silently substituting another boundary.
    """

    rows = get_features(WARD_LAYER)

    if not rows:
        raise RuntimeError(
            "The Birmingham Council ward service returned no features."
        )

    name_fields = [
        "WARDNME",
        "WARDNAME",
        "WARD_NAME",
        "NAME",
    ]

    ladywood_rows = []

    for row in rows:
        for field in name_fields:
            value = row.get(field)

            if (
                value is not None
                and str(value).strip().casefold() == "ladywood"
            ):
                ladywood_rows.append(row)
                break

    if not ladywood_rows:
        available_fields = sorted({
            key
            for row in rows
            for key in row.keys()
            if key != "geometry"
        })

        raise RuntimeError(
            "Ward features were returned, but Ladywood could not "
            "be identified. Available fields: "
            + ", ".join(available_fields)
        )

    geometries = []

    for row in ladywood_rows:
        geom = clean_geometry(row.get("geometry"))

        if geom is not None:
            geometries.append(geom)

    if not geometries:
        raise RuntimeError(
            "Ladywood was identified, but its boundary geometry "
            "was empty."
        )

    return unary_union(geometries)


# ============================================================
# LOAD LADYWOOD
# ============================================================

with st.spinner("Loading the official Ladywood boundary..."):
    try:
        ladywood_geometry = get_ladywood_boundary()
    except Exception as error:
        st.error(
            "The Ladywood boundary could not be loaded: "
            + str(error)
        )
        st.stop()


ladywood_bbox = ladywood_geometry.bounds


# ============================================================
# OFFICIAL LSOAs
# ============================================================

with st.spinner("Loading LSOA areas..."):
    try:
        lsoa_rows = get_features(
            LSOA_LAYER,
            geometry=ladywood_bbox
        )
    except Exception as error:
        st.error(
            "The official LSOA dataset could not be loaded: "
            + str(error)
        )
        st.stop()


lsoa_records = []

for row in lsoa_rows:
    geom = clean_geometry(row.get("geometry"))

    if geom is None:
        continue

    if not geom.intersects(ladywood_geometry):
        continue

    clipped = clean_geometry(
        geom.intersection(ladywood_geometry)
    )

    if clipped is None or clipped.area <= 0:
        continue

    lsoa_records.append({
        "LSOA21CD": row.get("LSOA21CD"),
        "LSOA21NM": row.get("LSOA21NM"),
        "MEAN": row.get("MEAN"),
        "MIN": row.get("MIN"),
        "MAX": row.get("MAX"),
        "MEDIAN": row.get("MEDIAN"),
        "AVERAGE_RISK": row.get("AVERAGE_RISK"),
        "MINIMUM_RISK": row.get("MINIMUM_RISK"),
        "MAXIMUM_RISK": row.get("MAXIMUM_RISK"),
        "geometry": clipped,
        "Area_m2": clipped.area,
    })


lsoa_df = pd.DataFrame(lsoa_records)

if lsoa_df.empty:
    st.error(
        "No LSOA polygons intersecting Ladywood were returned."
    )
    st.stop()


lsoa_df["CRVA_MEAN"] = pd.to_numeric(
    lsoa_df["MEAN"],
    errors="coerce"
)


# ============================================================
# ENVIRONMENTAL SPATIAL LAYERS
# ============================================================

def load_spatial_layer(layer_url, layer_name):
    """
    Retrieve a council layer and retain only geometries
    intersecting Ladywood. Query failures are reported separately.
    """

    try:
        rows = get_features(
            layer_url,
            geometry=ladywood_bbox
        )
    except Exception as error:
        st.warning(
            f"{layer_name} could not be retrieved: {error}"
        )
        return []

    features = []

    for row in rows:
        geom = clean_geometry(row.get("geometry"))

        if geom is None:
            continue

        if not geom.intersects(ladywood_geometry):
            continue

        clipped = clean_geometry(
            geom.intersection(ladywood_geometry)
        )

        if clipped is None:
            continue

        features.append({
            "attributes": row,
            "geometry": clipped,
        })

    return features


# Flood Zone 3 - restored as a separate layer
flood_zone_3_features = load_spatial_layer(
    FLOOD_ZONE_3_LAYER,
    "Flood Zone 3"
)

surface_features = load_spatial_layer(
    SURFACE_FLOOD_LAYER,
    "Surface-water flooding"
)

brownfield_features = load_spatial_layer(
    BROWNFIELD_LAYER,
    "Brownfield register"
)

road_features = load_spatial_layer(
    ROAD_LAYER,
    "Classified roads"
)


# ============================================================
# SURFACE-WATER FLOOD EXPOSURE
# ============================================================

def exposure_percent(lsoa_geometry, features):
    """
    Percentage of an LSOA polygon intersecting the supplied
    environmental extent. Returns None if the layer is unavailable.
    """

    if not features:
        return None

    intersections = []

    for feature in features:
        geom = feature.get("geometry")

        if geom is None:
            continue

        if not geom.intersects(lsoa_geometry):
            continue

        intersection = clean_geometry(
            geom.intersection(lsoa_geometry)
        )

        if intersection is not None:
            intersections.append(intersection)

    if not intersections:
        return 0.0

    combined = unary_union(intersections)
    total_area = lsoa_geometry.area

    if total_area <= 0:
        return None

    return (
        combined.area / total_area
    ) * 100


# ============================================================
# BUILD ENVIRONMENTAL RESULTS
# ============================================================

results = []

for _, row in lsoa_df.iterrows():
    geom = row["geometry"]

    surface_flood = exposure_percent(
        geom,
        surface_features
    )

    brownfield = exposure_percent(
        geom,
        brownfield_features
    )

    results.append({
        "LSOA21CD": row["LSOA21CD"],
        "LSOA21NM": row["LSOA21NM"],
        "Area_m2": row["Area_m2"],
        "CRVA_MEAN": row["CRVA_MEAN"],
        "CRVA_AVERAGE_RISK": row["AVERAGE_RISK"],
        "Surface_Flood_%": surface_flood,
        "Brownfield_Exposure_%": brownfield,
        "geometry": geom,
    })


risk_df = pd.DataFrame(results)


# ============================================================
# NORMALISE INDICATORS
# ============================================================

def normalise(series):
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() == 0:
        return pd.Series(
            float("nan"),
            index=series.index
        )

    minimum = numeric.min()
    maximum = numeric.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            float("nan"),
            index=series.index
        )

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (numeric - minimum) / (maximum - minimum)


risk_df["Climate_Index"] = normalise(
    risk_df["CRVA_MEAN"]
)

risk_df["Surface_Flood_Index"] = normalise(
    risk_df["Surface_Flood_%"]
)

risk_df["Brownfield_Index"] = normalise(
    risk_df["Brownfield_Exposure_%"]
)


# ============================================================
# ENVIRONMENTAL SCREENING INDEX
# ============================================================

# Equal weighting is a project-defined comparison method.
# It is NOT an official Birmingham Council risk formula.
#
# Missing indicators are not silently replaced with zero.

index_columns = [
    "Climate_Index",
    "Surface_Flood_Index",
    "Brownfield_Index",
]

risk_df["Indicators_Available"] = (
    risk_df[index_columns].notna().sum(axis=1)
)

risk_df["Environmental_Index"] = (
    risk_df[index_columns].mean(axis=1, skipna=True)
)


# ============================================================
# PROJECT-DEFINED SCREENING CLASS
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
    risk_df["Environmental_Index"].apply(classify_risk)
)


# ============================================================
# DOMINANT ENVIRONMENTAL DRIVER
# ============================================================

def dominant_driver(row):
    values = {
        "CRVA vulnerability": row["Climate_Index"],
        "Surface-water flood exposure": row["Surface_Flood_Index"],
        "Brownfield exposure": row["Brownfield_Index"],
    }

    valid = {
        key: value
        for key, value in values.items()
        if pd.notna(value)
    }

    if not valid:
        return "Not available"

    return max(valid, key=valid.get)


risk_df["Dominant_Environmental_Driver"] = risk_df.apply(
    dominant_driver,
    axis=1
)


# ============================================================
# CLASSIFIED ROAD PROXIMITY
# ============================================================

def nearest_road_info(lsoa_geometry, roads):
    if not roads:
        return None, "No classified road data"

    nearest_distance = None
    nearest_class = None

    for road in roads:
        geom = road.get("geometry")

        if geom is None:
            continue

        distance = lsoa_geometry.distance(geom)

        attributes = road.get("attributes", {})

        road_class = attributes.get("BRUM_CLASS")

        if (
            nearest_distance is None
            or distance < nearest_distance
        ):
            nearest_distance = distance
            nearest_class = road_class

    return nearest_distance, nearest_class


road_info = [
    nearest_road_info(row["geometry"], road_features)
    for _, row in risk_df.iterrows()
]

risk_df["Nearest_Classified_Road_m"] = [
    item[0] for item in road_info
]

risk_df["Nearest_Road_Class"] = [
    item[1] for item in road_info
]


# ============================================================
# MAP
# ============================================================

ladywood_center = transform(
    to_wgs84,
    ladywood_geometry.centroid
)

m = folium.Map(
    location=[ladywood_center.y, ladywood_center.x],
    zoom_start=14,
    control_scale=True,
    tiles="CartoDB positron"
)


# ------------------------------------------------------------
# MAP LEGEND
# ------------------------------------------------------------

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 12px;
border: 1px solid #777;
font-size: 13px;
box-shadow: 0 1px 5px rgba(0,0,0,0.3);
">
<b>Project screening class</b><br>
<span style="color:green;">●</span> Low<br>
<span style="color:orange;">●</span> Moderate<br>
<span style="color:red;">●</span> High<br>
<span style="color:grey;">●</span> Not available
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)


# ------------------------------------------------------------
# LADYWOOD BOUNDARY
# ------------------------------------------------------------

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
        "fillOpacity": 0,
    },
    tooltip="Ladywood Ward Boundary",
).add_to(m)


# ------------------------------------------------------------
# RISK COLOURS
# ------------------------------------------------------------

def risk_colour(value):
    if value == "HIGH":
        return "red"

    if value == "MODERATE":
        return "orange"

    if value == "LOW":
        return "green"

    return "gray"


# ------------------------------------------------------------
# LSOA SCREENING MAP
# ------------------------------------------------------------

lsoa_group = folium.FeatureGroup(
    name="LSOA Environmental Screening",
    show=True
)

for _, row in risk_df.iterrows():
    geom_wgs = transform(
        to_wgs84,
        row["geometry"]
    )

    risk = row["Risk_Class"]
    colour = risk_colour(risk)

    index_value = row["Environmental_Index"]

    index_text = (
        f"{index_value:.2f}"
        if pd.notna(index_value)
        else "Not available"
    )

    surface_value = row["Surface_Flood_%"]
    surface_text = (
        f"{surface_value:.1f}%"
        if pd.notna(surface_value)
        else "Unavailable"
    )

    brownfield_value = row["Brownfield_Exposure_%"]
    brownfield_text = (
        f"{brownfield_value:.1f}%"
        if pd.notna(brownfield_value)
        else "Unavailable"
    )

    popup = f"""
    <b>{row['LSOA21NM']}</b><br>
    LSOA code: {row['LSOA21CD']}<br>
    Project screening class: {risk}<br>
    Environmental index: {index_text}<br>
    CRVA MEAN: {row['CRVA_MEAN']}<br>
    Surface-water flood exposure: {surface_text}<br>
    Brownfield exposure: {brownfield_text}<br>
    Available indicators: {row['Indicators_Available']} of 3<br>
    Dominant indicator: {row['Dominant_Environmental_Driver']}
    """

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature, c=colour: {
            "color": c,
            "weight": 2,
            "fillColor": c,
            "fillOpacity": 0.25,
        },
        popup=folium.Popup(popup, max_width=400),
        tooltip=str(row["LSOA21NM"]),
    ).add_to(lsoa_group)

lsoa_group.add_to(m)


# ------------------------------------------------------------
# FLOOD ZONE 3 MAP SWITCH
# ------------------------------------------------------------

flood_zone_3_group = folium.FeatureGroup(
    name="Flood Zone 3 - High Probability",
    show=False
)

for feature in flood_zone_3_features:
    geom = feature.get("geometry")

    if geom is None:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "blue",
            "weight": 2,
            "fillColor": "blue",
            "fillOpacity": 0.35,
        },
        tooltip="Flood Zone 3 - High Probability",
    ).add_to(flood_zone_3_group)

flood_zone_3_group.add_to(m)


# ------------------------------------------------------------
# SURFACE-WATER FLOODING MAP SWITCH
# ------------------------------------------------------------

surface_group = folium.FeatureGroup(
    name="Surface-Water Flooding",
    show=False
)

for feature in surface_features:
    geom_wgs = transform(
        to_wgs84,
        feature["geometry"]
    )

    folium.GeoJson(
        mapping(geom_wgs),
        name="Surface-Water Flooding",
        style_function=lambda feature: {
            "color": "purple",
            "weight": 1,
            "fillColor": "purple",
            "fillOpacity": 0.25,
        },
        tooltip="Surface-Water Flooding",
    ).add_to(surface_group)

surface_group.add_to(m)


# ------------------------------------------------------------
# BROWNFIELD MAP SWITCH
# ------------------------------------------------------------

brownfield_group = folium.FeatureGroup(
    name="Brownfield Register Sites",
    show=False
)

for feature in brownfield_features:
    geom_wgs = transform(
        to_wgs84,
        feature["geometry"]
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "brown",
            "weight": 2,
            "fillColor": "brown",
            "fillOpacity": 0.25,
        },
        tooltip="Brownfield Register Site",
    ).add_to(brownfield_group)

brownfield_group.add_to(m)


# ------------------------------------------------------------
# CLASSIFIED ROADS MAP SWITCH
# ------------------------------------------------------------

road_group = folium.FeatureGroup(
    name="Classified Roads",
    show=False
)


def road_colour(road_class):
    text = str(road_class).strip().upper()

    if text.startswith("A"):
        return "red"

    if text.startswith("B"):
        return "blue"

    return "gold"


for feature in road_features:
    geom = feature.get("geometry")

    if geom is None:
        continue

    geom_wgs = transform(to_wgs84, geom)

    attributes = feature.get("attributes", {})
    road_class = attributes.get(
        "BRUM_CLASS",
        "Classified Road"
    )
    road_name = attributes.get("BRUM_ROAD_", "")

    tooltip = str(road_class)

    if road_name:
        tooltip += " — " + str(road_name)

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature, rc=road_class: {
            "color": road_colour(rc),
            "weight": 3,
        },
        tooltip=tooltip,
    ).add_to(road_group)

road_group.add_to(m)


# ------------------------------------------------------------
# LSOA CENTROID RISK MARKERS
# ------------------------------------------------------------

marker_group = folium.FeatureGroup(
    name="LSOA Screening Markers",
    show=True
)

for _, row in risk_df.iterrows():
    point = row["geometry"].centroid

    point_wgs = transform(
        to_wgs84,
        point
    )

    colour = risk_colour(row["Risk_Class"])

    folium.CircleMarker(
        location=[point_wgs.y, point_wgs.x],
        radius=7,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.9,
        tooltip=(
            f"{row['LSOA21NM']} — {row['Risk_Class']}"
        ),
        popup=folium.Popup(
            f"""
            <b>{row['LSOA21NM']}</b><br>
            Environmental index:
            {
                f"{row['Environmental_Index']:.2f}"
                if pd.notna(row['Environmental_Index'])
                else "Not available"
            }<br>
            Screening class: {row['Risk_Class']}<br>
            Dominant indicator:
            {row['Dominant_Environmental_Driver']}
            """,
            max_width=350,
        ),
    ).add_to(marker_group)

marker_group.add_to(m)


# ------------------------------------------------------------
# LAYER CONTROL
# ------------------------------------------------------------

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY MAP
# ============================================================

st.subheader("Ladywood Environmental Map")

st.caption(
    "Use the map layer control to display or hide Flood Zone 3, "
    "surface-water flooding, brownfield sites, classified roads "
    "and LSOA screening."
)

st.components.v1.html(
    m.get_root().render(),
    height=680,
    scrolling=False
)


# ============================================================
# ENVIRONMENTAL ASSESSMENT TABLE
# ============================================================

st.subheader("Environmental Screening Results")

st.caption(
    "The index is a project-defined comparative screening measure, "
    "not an official Birmingham Council risk classification."
)

table_columns = [
    "LSOA21CD",
    "LSOA21NM",
    "CRVA_MEAN",
    "CRVA_AVERAGE_RISK",
    "Surface_Flood_%",
    "Brownfield_Exposure_%",
    "Climate_Index",
    "Surface_Flood_Index",
    "Brownfield_Index",
    "Environmental_Index",
    "Indicators_Available",
    "Risk_Class",
    "Dominant_Environmental_Driver",
]

display_df = risk_df[table_columns].copy()

display_df = display_df.sort_values(
    "Environmental_Index",
    ascending=False,
    na_position="last"
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PRIORITY AREAS
# ============================================================

st.subheader("Areas for Further Investigation")

priority_df = risk_df[
    risk_df["Risk_Class"] == "HIGH"
].sort_values(
    "Environmental_Index",
    ascending=False
)

if priority_df.empty:
    st.info(
        "No LSOA reached the HIGH project screening category. "
        "This does not establish that environmental hazards are absent."
    )
else:
    st.dataframe(
        priority_df[
            [
                "LSOA21CD",
                "LSOA21NM",
                "Environmental_Index",
                "Indicators_Available",
                "Dominant_Environmental_Driver",
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# CHART 1 — ENVIRONMENTAL SCREENING INDEX
# ============================================================

st.subheader("Environmental Screening Index by Area")

index_chart = risk_df[
    ["LSOA21NM", "Environmental_Index"]
].set_index("LSOA21NM")

st.bar_chart(index_chart)


# ============================================================
# CHART 2 — CRVA INDICATOR
# ============================================================

st.subheader("CRVA MEAN by Area")

st.caption(
    "CRVA MEAN is displayed as supplied by the source. The "
    "normalised Climate Index is a relative comparison between "
    "the mapped LSOAs, not a separate official standard."
)

climate_chart = risk_df[
    ["LSOA21NM", "CRVA_MEAN"]
].set_index("LSOA21NM")

st.bar_chart(climate_chart)


# ============================================================
# CHART 3 — SURFACE-WATER FLOOD EXPOSURE
# ============================================================

st.subheader("Surface-Water Flood Exposure by Area")

st.caption(
    "Percentage of each LSOA polygon intersecting the retrieved "
    "surface-water flooding layer. Flood Zone 3 is displayed "
    "separately on the map."
)

surface_chart = risk_df[
    ["LSOA21NM", "Surface_Flood_%"]
].set_index("LSOA21NM")

st.bar_chart(surface_chart)


# ============================================================
# CHART 4 — BROWNFIELD EXPOSURE
# ============================================================

st.subheader("Brownfield Register Exposure by Area")

st.caption(
    "Percentage of each LSOA polygon intersecting mapped "
    "brownfield register sites. This does not represent all "
    "potentially degraded land."
)

brownfield_chart = risk_df[
    ["LSOA21NM", "Brownfield_Exposure_%"]
].set_index("LSOA21NM")

st.bar_chart(brownfield_chart)


# ============================================================
# CHART 5 — INDICATOR COMPARISON
# ============================================================

st.subheader("Comparison of Environmental Indicators")

driver_chart = risk_df[
    [
        "LSOA21NM",
        "Climate_Index",
        "Surface_Flood_Index",
        "Brownfield_Index",
    ]
].set_index("LSOA21NM")

st.bar_chart(driver_chart)


# ============================================================
# CLASSIFIED ROADS TABLE
# ============================================================

st.subheader("Classified Roads Within Ladywood")

st.caption(
    "Road proximity is presented as context. No arbitrary "
    "road-pollution multiplier is applied."
)

road_records = []

for feature in road_features:
    attributes = feature.get("attributes", {})

    road_records.append({
        "Road classification": attributes.get("BRUM_CLASS"),
        "Road": attributes.get("BRUM_ROAD_"),
    })

if road_records:
    roads_df = pd.DataFrame(road_records).drop_duplicates()

    sort_columns = [
        column for column in
        ["Road classification", "Road"]
        if column in roads_df.columns
    ]

    if sort_columns:
        roads_df = roads_df.sort_values(
            sort_columns,
            na_position="last"
        )

    st.dataframe(
        roads_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning(
        "No classified-road features were returned. "
        "This may indicate unavailable data or no returned features."
    )


# ============================================================
# ROAD PROXIMITY BY LSOA
# ============================================================

st.subheader("Classified Road Proximity by Area")

road_proximity_df = risk_df[
    [
        "LSOA21CD",
        "LSOA21NM",
        "Nearest_Classified_Road_m",
        "Nearest_Road_Class",
    ]
].copy()

road_proximity_df = road_proximity_df.sort_values(
    "Nearest_Classified_Road_m",
    na_position="last"
)

st.dataframe(
    road_proximity_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================

st.subheader("Engineering Interpretation")

st.write(
    "The dashboard uses mapped environmental indicators to identify "
    "areas that may warrant further investigation. The outputs support "
    "screening and prioritisation; they do not replace site inspections, "
    "environmental sampling, flood assessments or statutory evaluations."
)

st.write(
    "Flood Zone 3 is displayed as a separate map layer. The existing "
    "surface-water flooding indicator remains separate, so the "
    "dashboard does not combine the two flood datasets in its index."
)

st.write(
    "The equal-weight index and the HIGH/MODERATE/LOW thresholds are "
    "project-defined comparison rules. They must not be described as "
    "official Birmingham Council risk thresholds."
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
    "Surface_Flood_%",
    "Surface_Flood_Index",
    "Brownfield_Exposure_%",
    "Brownfield_Index",
    "Environmental_Index",
    "Indicators_Available",
    "Risk_Class",
    "Dominant_Environmental_Driver",
    "Nearest_Classified_Road_m",
    "Nearest_Road_Class",
]

download_df = risk_df[download_columns].copy()

st.download_button(
    label="Download Ladywood Environmental Assessment Results",
    data=download_df.to_csv(index=False).encode("utf-8"),
    file_name="Ladywood_Environmental_Assessment.csv",
    mime="text/csv"
)
