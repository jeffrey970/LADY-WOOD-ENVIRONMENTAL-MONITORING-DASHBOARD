import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium

from shapely.geometry import shape
from shapely.ops import unary_union


# ============================================================
# LADYWOOD ENVIRONMENTAL MONITORING DASHBOARD
# Official Birmingham City Council ArcGIS sources
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("LADYWOOD")
st.subheader("Environmental Monitoring Dashboard")
st.caption(
    "Official mapped environmental indicators for Ladywood, Birmingham. "
    "Mapped exposure is a screening indicator, not a property-level "
    "flood assessment or a prediction of future events."
)

# ------------------------------------------------------------
# OFFICIAL COUNCIL SOURCES
# ------------------------------------------------------------

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Boundaries/MapServer/0"
)

MYBRUM_ENV = (
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

LSOA_LAYER = f"{CRVA}/17"

# Try the primary CRVA layer first, then the alternative
# council-published MyMaps layer.
FLOOD_SOURCES = [
    (
        "CRVA 2025 — Flood Zone 3, layer 109",
        f"{CRVA}/109"
    ),
    (
        "MyMaps Environment — Flood Zone 3, layer 26",
        f"{MYBRUM_ENV}/26"
    ),
]

SURFACE_LAYER = f"{CRVA}/1546"
BROWNFIELD_LAYER = f"{HELAA}/38"
ROADS_LAYER = f"{HIGHWAYS}/26"

session = requests.Session()
session.headers.update(
    {"User-Agent": "LadywoodEnvironmentalDashboard/1.0"}
)


# ------------------------------------------------------------
# ARCGIS HELPERS
# ------------------------------------------------------------

def arcgis_json(url, params=None):
    """Request an ArcGIS REST JSON response and check for errors."""
    request_params = {"f": "json"}

    if params:
        request_params.update(params)

    response = session.get(
        url,
        params=request_params,
        timeout=90
    )
    response.raise_for_status()

    result = response.json()

    if "error" in result:
        error = result["error"]
        raise RuntimeError(
            error.get("message", str(error))
        )

    return result


def get_layer_features(layer_url, bbox=None):
    """
    Retrieve features by object IDs in batches.

    The ID query uses the service's native coordinate system,
    EPSG:27700. Feature geometries are requested as GeoJSON in
    EPSG:4326, then projected to EPSG:27700 for area calculations.
    """

    id_params = {
        "where": "1=1",
        "returnIdsOnly": "true",
        "returnGeometry": "false",
        "f": "json",
    }

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        id_params.update({
            "geometry": (
                f"{xmin},{ymin},{xmax},{ymax}"
            ),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "27700",
            "spatialRel": "esriSpatialRelIntersects",
        })

    id_result = arcgis_json(
        f"{layer_url}/query",
        id_params
    )

    object_ids = id_result.get("objectIds") or []

    if not object_ids:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    all_features = []
    batch_size = 200

    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start:start + batch_size]

        # GeoJSON output makes the returned geometry compatible
        # with Shapely's shape() function.
        feature_result = arcgis_json(
            f"{layer_url}/query",
            {
                "objectIds": ",".join(
                    str(item) for item in batch_ids
                ),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
            }
        )

        all_features.extend(
            feature_result.get("features", [])
        )

    rows = []
    geometries = []

    for feature in all_features:
        geometry = feature.get("geometry")

        if not geometry:
            continue

        try:
            geometries.append(shape(geometry))
            rows.append(feature.get("properties", {}))
        except Exception:
            continue

    if not geometries:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    result = gpd.GeoDataFrame(
        rows,
        geometry=geometries,
        crs="EPSG:4326"
    )

    return result.to_crs("EPSG:27700")


def clean_geometries(data):
    """Remove missing and empty geometries."""
    if data.empty:
        return data

    data = data[data.geometry.notna()].copy()
    data = data[~data.geometry.is_empty].copy()

    return data


def clip_to_boundary(data, boundary):
    """Clip features to the official Ladywood ward polygon."""
    if data.empty:
        return data

    data = data.to_crs("EPSG:27700")
    data = clean_geometries(data)

    if data.empty:
        return data

    ward_shape = unary_union(boundary.geometry)

    data["geometry"] = data.geometry.intersection(ward_shape)

    data = data[
        data.geometry.notna() & ~data.geometry.is_empty
    ].copy()

    return data


# ------------------------------------------------------------
# LADYWOOD BOUNDARY
# ------------------------------------------------------------

def get_ladywood_boundary():
    """
    Get the official ward polygons from Birmingham City Council's
    Internet_Boundaries service and select Ladywood by WARDNME.

    Filtering locally avoids relying on a SQL expression that may
    not be accepted by the service.
    """

    wards = get_layer_features(WARD_LAYER)

    if wards.empty:
        raise RuntimeError(
            "The council's Internet_Boundaries ward service "
            "returned no features."
        )

    name_column = None

    for column in wards.columns:
        if column.upper() == "WARDNME":
            name_column = column
            break

    if name_column is None:
        raise RuntimeError(
            "The ward service returned features, but the WARDNME "
            "field was not found. Returned columns: "
            + ", ".join(map(str, wards.columns))
        )

    names = (
        wards[name_column]
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    ladywood = wards[names == "ladywood"].copy()

    if ladywood.empty:
        available_names = sorted(
            wards[name_column].dropna().astype(str).unique()
        )

        raise RuntimeError(
            "The ward service returned features but no ward named "
            "'Ladywood' was found. Sample ward names: "
            + ", ".join(available_names[:15])
        )

    ladywood = ladywood.to_crs("EPSG:27700")
    ladywood = clean_geometries(ladywood)

    if ladywood.empty:
        raise RuntimeError(
            "Ladywood was identified, but its polygon geometry "
            "could not be loaded."
        )

    return ladywood


# ------------------------------------------------------------
# LOAD ALL DATA
# ------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data():

    diagnostics = {}

    # 1. Official ward boundary
    boundary = get_ladywood_boundary()

    xmin, ymin, xmax, ymax = boundary.total_bounds
    bbox = (xmin, ymin, xmax, ymax)

    # 2. LSOAs
    try:
        lsoas = get_layer_features(
            LSOA_LAYER,
            bbox=bbox
        )
        lsoas = clip_to_boundary(lsoas, boundary)
        diagnostics["LSOAs"] = (
            f"{len(lsoas)} feature(s) intersect Ladywood."
        )
    except Exception as exc:
        lsoas = gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )
        diagnostics["LSOAs"] = f"Query failed: {exc}"

    # 3. Surface flooding
    try:
        surface = get_layer_features(
            SURFACE_LAYER,
            bbox=bbox
        )
        surface = clip_to_boundary(surface, boundary)
        diagnostics["Surface flooding"] = (
            f"{len(surface)} feature(s) intersect Ladywood."
        )
    except Exception as exc:
        surface = gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )
        diagnostics["Surface flooding"] = f"Query failed: {exc}"

    # 4. Brownfield register
    try:
        brownfield = get_layer_features(
            BROWNFIELD_LAYER,
            bbox=bbox
        )
        brownfield = clip_to_boundary(
            brownfield,
            boundary
        )
        diagnostics["Brownfield"] = (
            f"{len(brownfield)} feature(s) intersect Ladywood."
        )
    except Exception as exc:
        brownfield = gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )
        diagnostics["Brownfield"] = f"Query failed: {exc}"

    # 5. Classified roads
    try:
        roads = get_layer_features(
            ROADS_LAYER,
            bbox=bbox
        )
        roads = clip_to_boundary(roads, boundary)
        diagnostics["Classified roads"] = (
            f"{len(roads)} feature(s) intersect Ladywood."
        )
    except Exception as exc:
        roads = gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )
        diagnostics["Classified roads"] = f"Query failed: {exc}"

    # 6. Flood Zone 3 — test both official council sources
    flood = gpd.GeoDataFrame(
        geometry=[],
        crs="EPSG:27700"
    )
    flood_source = None
    flood_diagnostics = []

    for source_label, source_url in FLOOD_SOURCES:
        try:
            candidate = get_layer_features(
                source_url,
                bbox=bbox
            )

            returned_count = len(candidate)

            if not candidate.empty:
                candidate = clip_to_boundary(
                    candidate,
                    boundary
                )

            intersect_count = len(candidate)

            flood_diagnostics.append(
                f"{source_label}: {returned_count} feature(s) "
                f"returned for the Ladywood bounding box; "
                f"{intersect_count} feature(s) intersect Ladywood."
            )

            if intersect_count > 0:
                flood = candidate
                flood_source = source_label
                break

        except Exception as exc:
            flood_diagnostics.append(
                f"{source_label}: query failed — {exc}"
            )

    if flood_source:
        diagnostics["Flood Zone 3"] = (
            f"Polygons retrieved from {flood_source}."
        )
    else:
        diagnostics["Flood Zone 3"] = (
            "Unavailable: neither source returned confirmed "
            "polygons intersecting Ladywood."
        )

    return (
        boundary,
        lsoas,
        flood,
        surface,
        brownfield,
        roads,
        flood_source,
        flood_diagnostics,
        diagnostics
    )


# ------------------------------------------------------------
# RUN DATA LOADING
# ------------------------------------------------------------

with st.spinner(
    "Loading official Birmingham Council map layers..."
):
    try:
        (
            boundary,
            lsoas,
            flood,
            surface,
            brownfield,
            roads,
            flood_source,
            flood_diagnostics,
            diagnostics
        ) = load_dashboard_data()

    except Exception as exc:
        st.error(
            "Dashboard data could not be loaded: "
            + str(exc)
        )
        st.info(
            "The dashboard stopped because the ward boundary "
            "could not be established. No environmental risk "
            "calculations have been made."
        )
        st.stop()


# ------------------------------------------------------------
# AREA CALCULATIONS
# ------------------------------------------------------------

def area_hectares(data):
    """
    Calculate the total mapped geometry area in hectares.
    Data must be in EPSG:27700.

    Overlapping features may overlap one another. Therefore,
    use a union to avoid counting the same ground repeatedly.
    """
    if data.empty:
        return 0.0

    geometry_union = unary_union(data.geometry)

    return geometry_union.area / 10000.0


ward_area_ha = area_hectares(boundary)

# These are mapped polygon areas, not official risk scores.
flood_area_ha = (
    area_hectares(flood) if flood_source else None
)

surface_area_ha = (
    area_hectares(surface) if not surface.empty else 0.0
)

brownfield_area_ha = (
    area_hectares(brownfield)
    if not brownfield.empty
    else 0.0
)


# ------------------------------------------------------------
# SUMMARY METRICS
# ------------------------------------------------------------

st.markdown("### Ladywood overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Ladywood ward area",
    f"{ward_area_ha:,.1f} ha"
)

col2.metric(
    "LSOAs mapped",
    str(len(lsoas))
)

col3.metric(
    "Flood Zone 3 area",
    (
        f"{flood_area_ha:,.2f} ha"
        if flood_area_ha is not None
        else "Unavailable"
    )
)

col4.metric(
    "Brownfield mapped area",
    f"{brownfield_area_ha:,.2f} ha"
)


# ------------------------------------------------------------
# FLOOD SOURCE DIAGNOSTICS
# ------------------------------------------------------------

with st.expander(
    "Flood Zone 3 source verification",
    expanded=(flood_source is None)
):
    for item in flood_diagnostics:
        st.write("• " + item)

    if flood_source:
        st.success(
            "Flood polygons intersecting Ladywood were retrieved "
            f"from: {flood_source}"
        )
        st.caption(
            "This confirms a mapped spatial intersection only. "
            "It is not a property-level flood-risk assessment."
        )
    else:
        st.warning(
            "Flood Zone 3 could not be confirmed from the tested "
            "sources. This does NOT mean that Ladywood has no "
            "flood risk. Flood Zone 3 exposure remains unavailable."
        )


# ------------------------------------------------------------
# INTERACTIVE MAP
# ------------------------------------------------------------

boundary_wgs = boundary.to_crs("EPSG:4326")
boundary_centre = unary_union(
    boundary_wgs.geometry
).centroid

map_view = folium.Map(
    location=[
        boundary_centre.y,
        boundary_centre.x
    ],
    zoom_start=13,
    tiles="CartoDB positron"
)


def add_polygon_layer(
    map_object,
    data,
    layer_name,
    colour,
    fill_opacity=0.35,
    line_weight=1
):
    """Add a polygon or line feature layer to the interactive map."""
    if data.empty:
        return

    mapped = data.to_crs("EPSG:4326")

    folium.GeoJson(
        mapped.__geo_interface__,
        name=layer_name,
        style_function=lambda feature: {
            "color": colour,
            "weight": line_weight,
            "fillColor": colour,
            "fillOpacity": fill_opacity,
        },
        tooltip=layer_name,
    ).add_to(map_object)


# Ladywood boundary
folium.GeoJson(
    boundary_wgs.__geo_interface__,
    name="Official Ladywood ward boundary",
    style_function=lambda feature: {
        "color": "#202020",
        "weight": 3,
        "fillOpacity": 0,
    },
    tooltip="Ladywood ward boundary",
).add_to(map_view)

# LSOAs, shown as a light outline
add_polygon_layer(
    map_view,
    lsoas,
    "LSOAs intersecting Ladywood",
    "#777777",
    fill_opacity=0.05,
    line_weight=1
)

# Environmental layers
add_polygon_layer(
    map_view,
    flood,
    "Flood Zone 3 (fluvial)",
    "#1677ff",
    fill_opacity=0.45,
    line_weight=1
)

add_polygon_layer(
    map_view,
    surface,
    "Surface flooding",
    "#26a6d1",
    fill_opacity=0.35,
    line_weight=1
)

add_polygon_layer(
    map_view,
    brownfield,
    "Brownfield register sites",
    "#8b5a2b",
    fill_opacity=0.45,
    line_weight=1
)

add_polygon_layer(
    map_view,
    roads,
    "Classified roads",
    "#666666",
    fill_opacity=0.15,
    line_weight=2
)

folium.LayerControl(
    collapsed=False
).add_to(map_view)

st.markdown("### Interactive environmental map")
st.caption(
    "Use the layer selector in the map to turn individual "
    "environmental layers on and off."
)

from streamlit_folium import st_folium

st_folium(
    map_view,
    use_container_width=True,
    height=600
)


# ------------------------------------------------------------
# DATA TABLES
# ------------------------------------------------------------

st.markdown("### Mapped environmental indicators")

summary_rows = [
    {
        "Indicator": "Official Ladywood ward boundary",
        "Mapped features": len(boundary),
        "Mapped area (ha)": round(ward_area_ha, 3),
        "Status": "Loaded"
    },
    {
        "Indicator": "LSOAs intersecting Ladywood",
        "Mapped features": len(lsoas),
        "Mapped area (ha)": (
            round(area_hectares(lsoas), 3)
            if not lsoas.empty
            else None
        ),
        "Status": (
            "Loaded" if not lsoas.empty
            else "No features returned"
        )
    },
    {
        "Indicator": "Flood Zone 3 (fluvial)",
        "Mapped features": (
            len(flood) if flood_source else None
        ),
        "Mapped area (ha)": (
            round(flood_area_ha, 3)
            if flood_area_ha is not None
            else None
        ),
        "Status": (
            "Loaded from " + flood_source
            if flood_source
            else "Unavailable — do not interpret as zero risk"
        )
    },
    {
        "Indicator": "Surface flooding",
        "Mapped features": len(surface),
        "Mapped area (ha)": round(surface_area_ha, 3),
        "Status": (
            "Loaded" if not surface.empty
            else "No features returned or query failed"
        )
    },
    {
        "Indicator": "Brownfield register sites",
        "Mapped features": len(brownfield),
        "Mapped area (ha)": round(brownfield_area_ha, 3),
        "Status": (
            "Loaded" if not brownfield.empty
            else "No features returned or query failed"
        )
    },
    {
        "Indicator": "Classified roads",
        "Mapped features": len(roads),
        "Mapped area (ha)": None,
        "Status": (
            "Loaded" if not roads.empty
            else "No features returned or query failed"
        )
    }
]

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# SOURCE DIAGNOSTICS
# ------------------------------------------------------------

with st.expander("Data-loading diagnostics"):
    for name, message in diagnostics.items():
        st.write(f"**{name}:** {message}")

    st.caption(
        "A zero feature count is a query result, not proof that "
        "the corresponding environmental hazard is absent."
    )


# ------------------------------------------------------------
# DOWNLOAD TABLE
# ------------------------------------------------------------

st.download_button(
    label="Download mapped indicators (CSV)",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name="ladywood_mapped_environmental_indicators.csv",
    mime="text/csv"
)


# ------------------------------------------------------------
# RESEARCH INTEGRITY NOTE
# ------------------------------------------------------------

st.markdown("### How to interpret these results")

st.write(
    "This dashboard displays mapped environmental features "
    "retrieved from Birmingham City Council services. Areas are "
    "calculated from returned geometries clipped to the official "
    "Ladywood boundary."
)

st.write(
    "The dashboard does not create an official risk score from "
    "these areas. No assumed 0.67 or 0.34 thresholds are applied. "
    "Missing data are marked unavailable rather than being treated "
    "as zero exposure."
)

st.write(
    "A brownfield register identifies registered sites; it is not "
    "a complete measurement of all degraded land. Surface-water "
    "flooding and Flood Zone 3 represent different mapped indicators "
    "and should not be treated as interchangeable."
)
