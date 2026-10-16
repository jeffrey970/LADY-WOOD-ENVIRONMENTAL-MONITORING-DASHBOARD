import streamlit as st
import pandas as pd
import requests
import folium

from shapely.geometry import (
    shape, Polygon, MultiPolygon, LineString,
    MultiLineString, Point, mapping
)
from shapely.ops import transform, unary_union
from pyproj import Transformer

# ============================================================
# PAGE AND SOURCES
# ============================================================
st.set_page_config(
    page_title="Ladywood Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham, UK Environmental Dashboard")

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

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

to_wgs84 = Transformer.from_crs(
    BNG, WGS84, always_xy=True
).transform

session = requests.Session()


# ============================================================
# ARCGIS GEOMETRY CONVERSION
# ============================================================
def esri_geometry_to_shape(geom):
    if not geom:
        return None

    if "rings" in geom:
        rings = geom.get("rings", [])
        polygons = []

        for ring in rings:
            if len(ring) >= 4:
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

        result = unary_union(polygons)
        return result if not result.is_empty else None

    if "paths" in geom:
        lines = []

        for path in geom.get("paths", []):
            if len(path) >= 2:
                try:
                    lines.append(LineString(path))
                except Exception:
                    continue

        if not lines:
            return None

        if len(lines) == 1:
            return lines[0]

        return MultiLineString(lines)

    if "x" in geom and "y" in geom:
        return Point(geom["x"], geom["y"])

    return None


# ============================================================
# ARCGIS QUERY
# ============================================================
def arcgis_request(
    layer_url,
    where="1=1",
    bbox=None,
    offset=None,
    ids_only=False
):
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "27700",
        "f": "json",
    }

    if ids_only:
        params["returnIdsOnly"] = "true"
        params.pop("outFields", None)
        params.pop("returnGeometry", None)

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox

        params["geometry"] = (
            '{"xmin":%s,"ymin":%s,"xmax":%s,"ymax":%s,'
            '"spatialReference":{"wkid":27700}}'
            % (xmin, ymin, xmax, ymax)
        )

        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"

    if offset is not None:
        params["resultOffset"] = offset
        params["resultRecordCount"] = 1000

    response = session.get(
        layer_url + "/query",
        params=params,
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(
            f"ArcGIS error for {layer_url}: {data['error']}"
        )

    return data


def get_features(layer_url, where="1=1", bbox=None):
    """
    Retrieve features from an ArcGIS layer.

    If the service indicates that the result exceeds its transfer
    limit, request subsequent pages.
    """
    first = arcgis_request(
        layer_url,
        where=where,
        bbox=bbox
    )

    features = first.get("features", [])

    if first.get("exceededTransferLimit", False):
        offset = len(features)

        while True:
            page = arcgis_request(
                layer_url,
                where=where,
                bbox=bbox,
                offset=offset
            )

            next_features = page.get("features", [])
            features.extend(next_features)

            if (
                not page.get("exceededTransferLimit", False)
                or not next_features
            ):
                break

            offset += len(next_features)

    output = []

    for feature in features:
        row = dict(feature.get("attributes", {}))
        row["geometry"] = esri_geometry_to_shape(
            feature.get("geometry")
        )
        output.append(row)

    return output


# ============================================================
# STANDARD SPATIAL LAYER LOADER
# ============================================================
def load_spatial_layer(
    layer_url,
    label,
    ward_geom,
    bbox
):
    """
    Load a layer using the Ladywood bounding box and clip features
    to the official Ladywood ward boundary.

    A failed request is reported separately from zero intersections.
    """
    try:
        rows = get_features(layer_url, bbox=bbox)

        features = []

        for row in rows:
            geom = row.get("geometry")

            if (
                geom is None
                or geom.is_empty
                or not geom.intersects(ward_geom)
            ):
                continue

            clipped = geom.intersection(ward_geom)

            if not clipped.is_empty:
                features.append({
                    "attributes": row,
                    "geometry": clipped
                })

        st.caption(
            f"{label}: {len(features)} feature(s) "
            "intersect Ladywood."
        )

        return features, None

    except Exception as exc:
        st.warning(f"Could not load {label}: {exc}")
        return [], str(exc)


# ============================================================
# SPECIAL FLOOD ZONE 3 LOADER
# ============================================================
def load_flood_zone_3(
    layer_url,
    label,
    ward_geom,
    bbox
):
    """
    Try the bounding-box query first.

    If no features are returned, retry without a spatial filter.
    Intersect the retrieved geometries with Ladywood locally.

    A successful query with zero intersections is distinguished
    from a query failure.
    """
    errors = []
    rows = []
    retrieval_method = None

    # Attempt 1: use Ladywood's bounding box.
    try:
        rows = get_features(
            layer_url,
            bbox=bbox
        )
        retrieval_method = "bounding-box query"

    except Exception as exc:
        errors.append(
            f"Bounding-box query failed: {exc}"
        )

    # Attempt 2: query without the bounding-box filter.
    if not rows:
        try:
            rows = get_features(layer_url)
            retrieval_method = "unfiltered query"

        except Exception as exc:
            errors.append(
                f"Unfiltered query failed: {exc}"
            )

    # If both attempts return no rows, do not claim that
    # Ladywood has no flood risk.
    if not rows:
        if errors:
            st.error(
                f"{label}: features could not be retrieved. "
                + " | ".join(errors)
            )
        else:
            st.warning(
                f"{label}: both queries returned zero source "
                "features. This result must be investigated; "
                "it does not establish that Ladywood has no "
                "Flood Zone 3 risk."
            )

        return [], "No features retrieved; query requires investigation"

    # Clip source geometries to the official Ladywood boundary.
    features = []

    for row in rows:
        geom = row.get("geometry")

        if geom is None or geom.is_empty:
            continue

        try:
            if not geom.intersects(ward_geom):
                continue

            clipped = geom.intersection(ward_geom)

            if not clipped.is_empty and clipped.area > 0:
                features.append({
                    "attributes": row,
                    "geometry": clipped
                })

        except Exception:
            continue

    st.caption(
        f"{label}: {len(features)} feature(s) intersect Ladywood. "
        f"Retrieval method: {retrieval_method}; "
        f"{len(rows)} source feature(s) retrieved."
    )

    # None means the query succeeded. The feature count may still
    # legitimately be zero after the local spatial intersection.
    return features, None


# ============================================================
# EXPOSURE CALCULATIONS
# ============================================================
def exposure_percent(
    area_geom,
    features,
    data_loaded=True
):
    """
    Calculate the percentage of an area intersecting the union
    of the supplied feature geometries.

    None represents unavailable data.
    Zero represents successfully loaded data with no overlap.
    """
    if not data_loaded:
        return None

    if not features:
        return 0.0

    intersections = []

    for feature in features:
        geom = feature.get("geometry")

        if (
            geom is not None
            and not geom.is_empty
            and geom.intersects(area_geom)
        ):
            intersection = geom.intersection(area_geom)

            if not intersection.is_empty:
                intersections.append(intersection)

    if not intersections or area_geom.area <= 0:
        return 0.0

    overlap = unary_union(intersections).area

    return 100 * overlap / area_geom.area


def normalise(series):
    numeric = pd.to_numeric(
        series,
        errors="coerce"
    )

    low = numeric.min()
    high = numeric.max()

    if (
        pd.isna(low)
        or pd.isna(high)
        or high == low
    ):
        return pd.Series(
            [
                0.0 if pd.notna(value) else float("nan")
                for value in numeric
            ],
            index=series.index
        )

    return (numeric - low) / (high - low)


def risk_colour(value):
    return {
        "HIGH": "red",
        "MODERATE": "orange",
        "LOW": "green"
    }.get(value, "gray")


# ============================================================
# OFFICIAL LADYWOOD WARD
# ============================================================
try:
    ward_rows = get_features(
        WARD_LAYER,
        where="WARDNME = 'Ladywood'"
    )

except Exception:
    ward_rows = []

if not ward_rows:
    try:
        all_wards = get_features(WARD_LAYER)

        ward_rows = [
            row for row in all_wards
            if str(
                row.get("WARDNME", "")
            ).strip().lower() == "ladywood"
        ]

    except Exception as exc:
        st.error(
            "Could not retrieve Birmingham ward boundaries: "
            f"{exc}"
        )
        st.stop()

if (
    not ward_rows
    or ward_rows[0].get("geometry") is None
):
    st.error(
        "The official Ladywood ward boundary could not "
        "be retrieved."
    )
    st.stop()

ladywood_geometry = ward_rows[0]["geometry"].buffer(0)

if ladywood_geometry.is_empty:
    st.error(
        "The official Ladywood ward geometry is empty."
    )
    st.stop()

ladywood_bbox = ladywood_geometry.bounds


# ============================================================
# OFFICIAL LSOAs CLIPPED TO LADYWOOD
# ============================================================
try:
    lsoa_rows = get_features(
        LSOA_LAYER,
        bbox=ladywood_bbox
    )

except Exception as exc:
    st.error(
        f"The official LSOA dataset could not be loaded: {exc}"
    )
    st.stop()

lsoa_records = []

for row in lsoa_rows:
    geom = row.get("geometry")

    if geom is None or geom.is_empty:
        continue

    clipped = geom.intersection(ladywood_geometry)

    if clipped.is_empty or clipped.area <= 0:
        continue

    lsoa_records.append({
        "LSOA21CD": row.get("LSOA21CD"),
        "LSOA21NM": row.get("LSOA21NM"),
        "MEAN": row.get("MEAN"),
        "AVERAGE_RISK": row.get("AVERAGE_RISK"),
        "MINIMUM_RISK": row.get("MINIMUM_RISK"),
        "MAXIMUM_RISK": row.get("MAXIMUM_RISK"),
        "geometry": clipped,
        "Area_m2": clipped.area
    })

lsoa_df = pd.DataFrame(lsoa_records)

if lsoa_df.empty:
    st.error(
        "No official LSOA areas intersecting Ladywood were found."
    )
    st.stop()

lsoa_df["CRVA_MEAN"] = pd.to_numeric(
    lsoa_df["MEAN"],
    errors="coerce"
)


# ============================================================
# LOAD ENVIRONMENTAL DATA
# ============================================================
st.subheader("Environmental data loading")

# IMPORTANT: use the special loader for Flood Zone 3.
flood3_features, flood3_error = load_flood_zone_3(
    FLOOD_ZONE_3_LAYER,
    "Flood Zone 3 (fluvial, high probability)",
    ladywood_geometry,
    ladywood_bbox
)

surface_features, surface_error = load_spatial_layer(
    SURFACE_FLOOD_LAYER,
    "Surface flooding",
    ladywood_geometry,
    ladywood_bbox
)

brownfield_features, brownfield_error = load_spatial_layer(
    BROWNFIELD_LAYER,
    "Brownfield register",
    ladywood_geometry,
    ladywood_bbox
)

road_features, road_error = load_spatial_layer(
    ROAD_LAYER,
    "Classified roads",
    ladywood_geometry,
    ladywood_bbox
)


# ============================================================
# CALCULATE EXPOSURE BY LSOA
# ============================================================
results = []

for _, row in lsoa_df.iterrows():
    geom = row["geometry"]

    fz3 = exposure_percent(
        geom,
        flood3_features,
        data_loaded=(flood3_error is None)
    )

    surface = exposure_percent(
        geom,
        surface_features,
        data_loaded=(surface_error is None)
    )

    brown = exposure_percent(
        geom,
        brownfield_features,
        data_loaded=(brownfield_error is None)
    )

    valid_flood = [
        value for value in (fz3, surface)
        if pd.notna(value)
    ]

    combined_flood = (
        max(valid_flood)
        if valid_flood
        else float("nan")
    )

    results.append({
        "LSOA21CD": row["LSOA21CD"],
        "LSOA21NM": row["LSOA21NM"],
        "Area_m2": row["Area_m2"],
        "CRVA_MEAN": row["CRVA_MEAN"],
        "CRVA_AVERAGE_RISK": row["AVERAGE_RISK"],
        "Flood_Zone_3_%": fz3,
        "Surface_Flood_%": surface,
        "Combined_Flood_Exposure_%": combined_flood,
        "Brownfield_Exposure_%": brown,
        "geometry": geom
    })

risk_df = pd.DataFrame(results)

risk_df["Climate_Index"] = normalise(
    risk_df["CRVA_MEAN"]
)

risk_df["Flood_Index"] = normalise(
    risk_df["Combined_Flood_Exposure_%"]
)

risk_df["Brownfield_Index"] = normalise(
    risk_df["Brownfield_Exposure_%"]
)

# Project-defined equal-weight screening index.
# These weights and cutoffs are not official statutory ratings.
valid_components = risk_df[
    [
        "Climate_Index",
        "Flood_Index",
        "Brownfield_Index"
    ]
]

risk_df["Environmental_Index"] = valid_components.mean(
    axis=1,
    skipna=False
)


def classify_risk(value):
    if pd.isna(value):
        return "NOT AVAILABLE"

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


risk_df["Risk_Class"] = risk_df[
    "Environmental_Index"
].apply(classify_risk)


def dominant_driver(row):
    values = {
        "Climate vulnerability": row["Climate_Index"],
        "Flood exposure": row["Flood_Index"],
        "Brownfield exposure": row["Brownfield_Index"]
    }

    values = {
        key: value
        for key, value in values.items()
        if pd.notna(value)
    }

    return (
        max(values, key=values.get)
        if values
        else "Not available"
    )


risk_df["Dominant_Environmental_Driver"] = risk_df.apply(
    dominant_driver,
    axis=1
)


# ============================================================
# NEAREST CLASSIFIED ROAD
# ============================================================
def nearest_road_info(area_geom, roads):
    nearest_distance = None
    nearest_class = None

    for road in roads:
        geom = road.get("geometry")

        if geom is None or geom.is_empty:
            continue

        distance = area_geom.distance(geom)

        if (
            nearest_distance is None
            or distance < nearest_distance
        ):
            nearest_distance = distance

            nearest_class = road.get(
                "attributes", {}
            ).get("BRUM_CLASS")

    return nearest_distance, nearest_class


road_distances = [
    nearest_road_info(row["geometry"], road_features)
    for _, row in risk_df.iterrows()
]

risk_df["Nearest_Classified_Road_m"] = [
    item[0] for item in road_distances
]

risk_df["Nearest_Road_Class"] = [
    item[1] for item in road_distances
]


# ============================================================
# MAP BASE AND LADYWOOD BOUNDARY
# ============================================================
center = transform(
    to_wgs84,
    ladywood_geometry.centroid
)

m = folium.Map(
    location=[center.y, center.x],
    zoom_start=14,
    control_scale=True,
    tiles="OpenStreetMap"
)

ladywood_wgs = transform(
    to_wgs84,
    ladywood_geometry
)

folium.GeoJson(
    mapping(ladywood_wgs),
    name="Ladywood Boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 3,
        "fillOpacity": 0
    },
    tooltip="Official Ladywood Ward Boundary"
).add_to(m)

legend_html = """
<div style="
position:fixed;
bottom:30px;
left:30px;
z-index:9999;
background:white;
padding:12px;
border:1px solid #777;
font-size:13px;
box-shadow:0 1px 5px rgba(0,0,0,.3);
">
<b>Environmental Screening</b><br>
<span style="color:green">●</span> Low<br>
<span style="color:orange">●</span> Moderate<br>
<span style="color:red">●</span> High<br>
<span style="color:grey">●</span> Not available<br>
<span style="color:#00BFFF">■</span> Flood Zone 3
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)


# ============================================================
# LSOA SCREENING POLYGONS
# ============================================================
lsoa_group = folium.FeatureGroup(
    name="Ladywood LSOA Risk",
    show=True
)

for _, row in risk_df.iterrows():
    geom_wgs = transform(
        to_wgs84,
        row["geometry"]
    )

    colour = risk_colour(row["Risk_Class"])
    env_idx = row["Environmental_Index"]
    flood_val = row["Combined_Flood_Exposure_%"]

    if pd.notna(env_idx):
        popup = (
            f"<b>{row['LSOA21NM']}</b><br>"
            f"LSOA: {row['LSOA21CD']}<br>"
            f"Screening class: {row['Risk_Class']}<br>"
            f"Environmental index: {env_idx:.2f}"
        )
    else:
        popup = (
            f"<b>{row['LSOA21NM']}</b><br>"
            f"LSOA: {row['LSOA21CD']}<br>"
            f"Screening class: {row['Risk_Class']}<br>"
            "Environmental index: Not available"
        )

    if pd.notna(row["Climate_Index"]):
        popup += (
            f"<br>Climate vulnerability index: "
            f"{row['Climate_Index']:.2f}"
        )
    else:
        popup += (
            "<br>Climate vulnerability index: Not available"
        )

    if pd.notna(flood_val):
        popup += f"<br>Flood exposure: {flood_val:.1f}%"
    else:
        popup += "<br>Flood exposure: Not available"

    if pd.notna(row["Brownfield_Exposure_%"]):
        popup += (
            f"<br>Brownfield exposure: "
            f"{row['Brownfield_Exposure_%']:.1f}%"
        )
    else:
        popup += "<br>Brownfield exposure: Not available"

    popup += (
        f"<br>Dominant driver: "
        f"{row['Dominant_Environmental_Driver']}"
    )

    distance = row["Nearest_Classified_Road_m"]

    if pd.notna(distance):
        popup += (
            f"<br>Nearest classified road: {distance:.0f} m"
        )
    else:
        popup += "<br>Nearest classified road: Not available"

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature, c=colour: {
            "color": c,
            "weight": 2,
            "fillColor": c,
            "fillOpacity": 0.20
        },
        popup=folium.Popup(
            popup,
            max_width=400
        )
    ).add_to(lsoa_group)

lsoa_group.add_to(m)


# ============================================================
# FLOOD ZONE 3 MAP LAYER
# ============================================================
flood_group = folium.FeatureGroup(
    name="Flood Zone 3",
    show=False,
    overlay=True,
    control=True
)

for feature in flood3_features:
    geom = feature.get("geometry")

    if geom is None or geom.is_empty:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "#0033CC",
            "weight": 2.5,
            "fillColor": "#00BFFF",
            "fillOpacity": 0.65,
            "opacity": 1.0
        },
        tooltip=(
            "Flood Zone 3 — high-probability "
            "fluvial flood zone"
        )
    ).add_to(flood_group)

flood_group.add_to(m)


# ============================================================
# SURFACE FLOODING MAP LAYER
# ============================================================
surface_group = folium.FeatureGroup(
    name="Surface Flooding",
    show=False
)

for feature in surface_features:
    geom = feature.get("geometry")

    if geom is None or geom.is_empty:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "purple",
            "weight": 1.5,
            "fillColor": "purple",
            "fillOpacity": 0.25
        },
        tooltip="Surface Flooding"
    ).add_to(surface_group)

surface_group.add_to(m)


# ============================================================
# BROWNFIELD MAP LAYER
# ============================================================
brown_group = folium.FeatureGroup(
    name="Brownfield Register",
    show=False
)

for feature in brownfield_features:
    geom = feature.get("geometry")

    if geom is None or geom.is_empty:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature: {
            "color": "#7B3F00",
            "weight": 2,
            "fillColor": "#A0522D",
            "fillOpacity": 0.30
        },
        tooltip="Brownfield Register"
    ).add_to(brown_group)

brown_group.add_to(m)


# ============================================================
# CLASSIFIED ROADS MAP LAYER
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
    geom = feature.get("geometry")

    if geom is None or geom.is_empty:
        continue

    geom_wgs = transform(
        to_wgs84,
        geom
    )

    attributes = feature.get("attributes", {})

    road_class = attributes.get(
        "BRUM_CLASS",
        "Classified Road"
    )

    road_name = attributes.get(
        "BRUM_ROAD_",
        ""
    )

    tooltip = road_class

    if road_name:
        tooltip += f" — {road_name}"

    folium.GeoJson(
        mapping(geom_wgs),
        style_function=lambda feature, rc=road_class: {
            "color": road_colour(rc),
            "weight": 3
        },
        tooltip=tooltip
    ).add_to(road_group)

road_group.add_to(m)


# ============================================================
# RISK MARKERS
# ============================================================
marker_group = folium.FeatureGroup(
    name="Risk Markers",
    show=True
)

for _, row in risk_df.iterrows():
    point = transform(
        to_wgs84,
        row["geometry"].centroid
    )

    colour = risk_colour(row["Risk_Class"])

    folium.CircleMarker(
        location=[point.y, point.x],
        radius=7,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.9,
        tooltip=(
            f"{row['LSOA21NM']} — {row['Risk_Class']}"
        ),
        popup=(
            f"<b>{row['LSOA21NM']}</b><br>"
            f"Risk: {row['Risk_Class']}<br>"
            f"Main driver: "
            f"{row['Dominant_Environmental_Driver']}"
        )
    ).add_to(marker_group)

marker_group.add_to(m)

folium.LayerControl(
    collapsed=False
).add_to(m)


# ============================================================
# DISPLAY MAP
# ============================================================
st.subheader("Ladywood Environmental Map")

st.caption(
    "Use the layer control in the top-right of the map "
    "to turn layers on and off. Flood Zone 3 is shown "
    "in bright blue when enabled."
)

if flood3_error is not None:
    st.warning(
        "Flood Zone 3 could not be verified from the source "
        "query. Its exposure is marked unavailable rather "
        "than being interpreted as zero."
    )
elif not flood3_features:
    st.warning(
        "The Flood Zone 3 query returned source features, "
        "but none intersected the official Ladywood boundary. "
        "Verify the service extent and spatial reference "
        "before interpreting this as no exposure."
    )

st.components.v1.html(
    m.get_root().render(),
    height=680,
    scrolling=False
)


# ============================================================
# ENVIRONMENTAL ASSESSMENT TABLE
# ============================================================
st.subheader("Environmental Assessment Classification")

st.caption(
    "This table compares official Ladywood LSOA areas using "
    "climate vulnerability, flood exposure and brownfield "
    "exposure. The combined index is a project-defined "
    "screening measure, not an official statutory rating."
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

display_df = risk_df[table_columns].sort_values(
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
st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

priority_df = risk_df[
    risk_df["Risk_Class"] == "HIGH"
].sort_values(
    "Environmental_Index",
    ascending=False
)

if priority_df.empty:
    st.info(
        "No Ladywood LSOA reached the HIGH assessment "
        "category using this project screening index."
    )

    moderate_df = risk_df[
        risk_df["Risk_Class"] == "MODERATE"
    ].sort_values(
        "Environmental_Index",
        ascending=False
    )

    if not moderate_df.empty:
        st.write("**Highest MODERATE assessment areas:**")

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
# ENVIRONMENTAL RISK CHARTS
# ============================================================
st.subheader("Environmental Risk by Ladywood Area")

st.caption(
    "Higher values indicate higher combined project screening "
    "scores. The index is not an official statutory risk rating."
)

st.bar_chart(
    risk_df.set_index("LSOA21NM")[
        ["Environmental_Index"]
    ]
)


st.subheader("Climate Vulnerability by Ladywood Area")

st.caption(
    "Numeric MEAN value retrieved from Birmingham's "
    "2025 LSOA environmental-risk dataset."
)

st.bar_chart(
    risk_df.set_index("LSOA21NM")[
        ["CRVA_MEAN"]
    ]
)


st.subheader("Flood Exposure by Ladywood Area")

st.caption(
    "Percentage of each LSOA intersecting the available "
    "combined flood assessment extent. Unavailable data "
    "are not treated as zero."
)

st.bar_chart(
    risk_df.set_index("LSOA21NM")[
        ["Combined_Flood_Exposure_%"]
    ]
)


st.subheader("Brownfield Exposure by Ladywood Area")

st.caption(
    "Percentage of each LSOA intersecting the Birmingham "
    "Brownfield Register layer."
)

st.bar_chart(
    risk_df.set_index("LSOA21NM")[
        ["Brownfield_Exposure_%"]
    ]
)


st.subheader("Environmental Drivers by Ladywood Area")

st.caption(
    "Normalised indicators compare the available environmental "
    "factors on a 0–1 scale. These values are relative to the "
    "areas included in this dataset."
)

st.bar_chart(
    risk_df.set_index("LSOA21NM")[
        [
            "Climate_Index",
            "Flood_Index",
            "Brownfield_Index"
        ]
    ]
)


# ============================================================
# CLASSIFIED ROADS TABLE
# ============================================================
st.subheader("Classified Roads Within Ladywood")

st.caption(
    "Official Birmingham classified-road data are shown "
    "as environmental context. Road proximity is reported "
    "separately; no arbitrary pollution multiplier is applied."
)

road_records = [
    {
        "Road classification": feature.get(
            "attributes", {}
        ).get("BRUM_CLASS"),
        "Road": feature.get(
            "attributes", {}
        ).get("BRUM_ROAD_")
    }
    for feature in road_features
]

if road_records:
    roads_df = pd.DataFrame(
        road_records
    ).drop_duplicates()

    roads_df = roads_df.sort_values(
        ["Road classification", "Road"],
        na_position="last"
    )

    st.dataframe(
        roads_df,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning(
        "No classified-road features were returned inside Ladywood."
    )


st.subheader("Classified Road Proximity by Ladywood Area")

st.dataframe(
    risk_df[
        [
            "LSOA21CD",
            "LSOA21NM",
            "Nearest_Classified_Road_m",
            "Nearest_Road_Class"
        ]
    ].sort_values(
        "Nearest_Classified_Road_m",
        na_position="last"
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================
st.subheader("Engineering Assessment Interpretation")

st.caption(
    "The dashboard applies a source → pathway → receptor "
    "screening approach. Indicators identify areas where "
    "further engineering investigation may be useful. "
    "They do not replace site investigation, official flood "
    "risk assessment or statutory environmental assessment."
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================
st.subheader("Download Assessment Data")

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

download_df = risk_df[download_columns].copy()

st.download_button(
    label="Download Ladywood Environmental Assessment Results",
    data=download_df.to_csv(index=False),
    file_name="Ladywood_Environmental_Assessment.csv",
    mime="text/csv"
)
