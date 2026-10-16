import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium

from shapely.geometry import shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

# ============================================================
# LADYWOOD ENVIRONMENTAL MONITORING DASHBOARD
# Data sources: Birmingham City Council ArcGIS REST services
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Dashboard",
    layout="wide"
)

st.title("LADYWOOD")
st.subheader("Environmental Monitoring Dashboard")
st.caption(
    "Official mapped indicators for Ladywood, Birmingham. "
    "Mapped exposure is a screening indicator, not a prediction "
    "of an individual property's risk."
)

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)
MYBRUM = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Environment/MapServer"
)
HELAA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer"
)
HIGHWAYS = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Highways_open/MapServer"
)

# Official layer IDs
WARD_LAYER = f"{CRVA}/14"
LSOA_LAYER = f"{CRVA}/17"
FLOOD_SOURCES = [
    ("CRVA 2025: Flood Zone 3 (layer 109)", f"{CRVA}/109"),
    ("MyMaps Environment: Flood Zone 3 (layer 26)", f"{MYBRUM}/26"),
]
SURFACE_LAYER = f"{CRVA}/1546"
BROWNFIELD_LAYER = f"{HELAA}/38"
ROADS_LAYER = f"{HIGHWAYS}/26"

session = requests.Session()
session.headers.update({"User-Agent": "LadywoodEnvironmentalDashboard/1.0"})


def arcgis_json(url, params=None):
    """Request ArcGIS JSON and report meaningful errors."""
    p = {"f": "json"}
    if params:
        p.update(params)

    response = session.get(url, params=p, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))

    return data


def get_layer_features(layer_url, where="1=1", bbox=None):
    """
    Fetch all feature pages. Use IDs first to avoid relying on a
    single response being below the service's record limit.
    Returns a GeoDataFrame in British National Grid (EPSG:27700).
    """
    params = {
        "where": where,
        "returnIdsOnly": "true",
        "returnGeometry": "false",
        "f": "json",
    }
    if bbox:
        params["geometry"] = (
            f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        )
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"

    id_result = arcgis_json(f"{layer_url}/query", params)
    object_ids = id_result.get("objectIds") or []

    if not object_ids:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    features = []
    batch_size = 200

    for start in range(0, len(object_ids), batch_size):
        batch = object_ids[start:start + batch_size]
        result = arcgis_json(
            f"{layer_url}/query",
            {
                "objectIds": ",".join(map(str, batch)),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "27700",
                "f": "json",
            },
        )
        features.extend(result.get("features", []))

    records = []
    geometries = []

    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue
        try:
            geometries.append(shape(geom))
            records.append(feature.get("attributes", {}))
        except Exception:
            continue

    if not geometries:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    return gpd.GeoDataFrame(
        records,
        geometry=geometries,
        crs="EPSG:27700"
    )


def get_ladywood_boundary():
    """Retrieve the official ward polygon and identify Ladywood."""
    layer_info = arcgis_json(WARD_LAYER)
    field_names = [
        field["name"] for field in layer_info.get("fields", [])
    ]

    candidates = [
        name for name in field_names
        if name.lower() in ("wardname", "ward_name", "wardnme", "name")
    ]

    # Query common field names individually; use exact match first.
    for field in candidates + ["WARDNME", "WARDNAME", "NAME"]:
        try:
            data = get_layer_features(
                WARD_LAYER,
                where=f"UPPER({field}) = 'LADYWOOD'"
            )
            if not data.empty:
                return data
        except Exception:
            continue

    # Fallback: retrieve all wards and filter locally.
    wards = get_layer_features(WARD_LAYER)
    if wards.empty:
        raise RuntimeError("The official ward layer returned no features.")

    for column in wards.columns:
        if column.lower() in (
            "wardnme", "wardname", "ward_name", "name"
        ):
            match = wards[column].astype(str).str.strip().str.upper()
            selected = wards[match == "LADYWOOD"]
            if not selected.empty:
                return selected

    raise RuntimeError(
        "Could not identify Ladywood in the official ward layer. "
        f"Returned fields: {list(wards.columns)}"
    )


def clip_to_boundary(data, boundary):
    """Clip returned features to the official Ladywood boundary."""
    if data.empty:
        return data

    boundary_union = unary_union(boundary.geometry)
    data = data[data.geometry.notna()].copy()
    data = data[~data.geometry.is_empty].copy()

    if data.empty:
        return data

    data["geometry"] = data.geometry.intersection(boundary_union)
    data = data[
        data.geometry.notna() & ~data.geometry.is_empty
    ].copy()

    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data():
    boundary = get_ladywood_boundary()

    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:27700")
    else:
        boundary = boundary.to_crs("EPSG:27700")

    # Boundary bounding box limits unnecessary downloads.
    minx, miny, maxx, maxy = boundary.total_bounds
    bbox = (minx, miny, maxx, maxy)

    lsoas = get_layer_features(LSOA_LAYER, bbox=bbox)
    lsoas = clip_to_boundary(lsoas, boundary)

    surface = get_layer_features(SURFACE_LAYER, bbox=bbox)
    surface = clip_to_boundary(surface, boundary)

    brownfield = get_layer_features(BROWNFIELD_LAYER, bbox=bbox)
    brownfield = clip_to_boundary(brownfield, boundary)

    roads = get_layer_features(ROADS_LAYER, bbox=bbox)
    roads = clip_to_boundary(roads, boundary)

    # Try both council-published Flood Zone 3 services.
    flood = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
    flood_source = None
    flood_diagnostics = []

    for label, url in FLOOD_SOURCES:
        try:
            candidate = get_layer_features(url, bbox=bbox)
            count_before_clip = len(candidate)

            if not candidate.empty:
                candidate = clip_to_boundary(candidate, boundary)

            count_after_clip = len(candidate)
            flood_diagnostics.append(
                f"{label}: {count_before_clip} feature(s) "
                f"returned in the bounding box; "
                f"{count_after_clip} intersect Ladywood."
            )

            if count_after_clip > 0:
                flood = candidate
                flood_source = label
                break

        except Exception as exc:
            flood_diagnostics.append(f"{label}: query failed ({exc}).")

    return (
        boundary, lsoas, flood, surface, brownfield, roads,
        flood_source, flood_diagnostics
    )


with st.spinner("Loading official Birmingham Council map layers..."):
    try:
        (
            boundary, lsoas, flood, surface, brownfield, roads,
            flood_source, flood_diagnostics
        ) = load_dashboard_data()
    except Exception as exc:
        st.error(f"Dashboard data could not be loaded: {exc}")
        st.stop()


# -------------------- SUMMARY METRICS --------------------

def area_hectares(data):
    if data.empty:
        return 0.0
    return unary_union(data.geometry).area / 10_000


c1, c2, c3, c4 = st.columns(4)
c1.metric("Mapped LSOAs", len(lsoas))
c2.metric("Flood Zone 3 area mapped (ha)", f"{area_hectares(flood):,.2f}")
c3.metric("Surface flooding mapped (ha)", f"{area_hectares(surface):,.2f}")
c4.metric("Brownfield area mapped (ha)", f"{area_hectares(brownfield):,.2f}")


# -------------------- FLOOD SOURCE STATUS --------------------

with st.expander("Flood Zone 3 source verification", expanded=not bool(flood_source)):
    for diagnostic in flood_diagnostics:
        st.write("• " + diagnostic)

    if flood_source:
        st.success(f"Flood polygons retrieved from: {flood_source}")
        st.caption(
            "The source returned polygons intersecting Ladywood. "
            "This confirms mapped overlap only; it does not replace "
            "a property-level flood assessment."
        )
    else:
        st.warning(
            "Flood Zone 3 could not be confirmed from either query. "
            "Do not interpret this as zero flood risk. Flood Zone 3 "
            "exposure remains unavailable until the query is resolved."
        )


# -------------------- MAP --------------------

boundary_wgs = boundary.to_crs("EPSG:4326")
center = boundary_wgs.geometry.union_all().centroid
m = folium.Map(
    location=[center.y, center.x],
    zoom_start=13,
    tiles="CartoDB positron"
)

folium.GeoJson(
    boundary_wgs,
    name="Official Ladywood boundary",
    style_function=lambda feature: {
        "color": "#222222",
        "weight": 3,
        "fill": False,
    },
    tooltip="Ladywood ward boundary",
).add_to(m)


def add_layer(data, label, colour, fill_opacity=0.35):
    if data.empty:
        return

    mapped = data.to_crs("EPSG:4326")

    folium.GeoJson(
        mapped,
        name=label,
        style_function=lambda feature: {
            "color": colour,
            "weight": 1,
            "fillColor": colour,
            "fillOpacity": fill_opacity,
        },
        tooltip=label,
    ).add_to(m)


add_layer(flood, "Flood Zone 3 (fluvial)", "#1677ff")
add_layer(surface, "Surface flooding (pluvial)", "#26a6d1")
add_layer(brownfield, "Brownfield register sites", "#8b5a2b")
add_layer(roads, "Classified roads", "#777777", 0.15)

folium.LayerControl(collapsed=False).add_to(m)
st.subheader("Ladywood environmental map")
st_folium(m, use_container_width=True, height=600)


# -------------------- DATA TABLES --------------------

st.subheader("Mapped feature counts")
summary = pd.DataFrame([
    {"Indicator": "Official Ladywood boundary", "Features": len(boundary)},
    {"Indicator": "LSOAs intersecting Ladywood", "Features": len(lsoas)},
    {"Indicator": "Flood Zone 3 polygons intersecting Ladywood",
     "Features": len(flood) if flood_source else None},
    {"Indicator": "Surface flooding features", "Features": len(surface)},
    {"Indicator": "Brownfield register features", "Features": len(brownfield)},
    {"Indicator": "Classified road features", "Features": len(roads)},
])
st.dataframe(summary, use_container_width=True, hide_index=True)

st.caption(
    "Source status: an empty query is not evidence of no hazard. "
    "Mapped area is calculated from the geometries returned by the "
    "council services. The map layers may overlap, so their areas "
    "must not be added together as if they were separate land parcels."
)

# Download the source-feature counts for your project records.
st.download_button(
    "Download mapped feature counts (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="ladywood_mapped_feature_counts.csv",
    mime="text/csv",
)
