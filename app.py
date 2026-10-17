import base64
from io import BytesIO

import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
import folium
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
from shapely.geometry import shape

# Ladywood Environmental Monitoring Dashboard
# Data sources are official Birmingham City Council ArcGIS services.
# This is a screening dashboard, not a statutory risk assessment.
# It does not invent missing values or claim that background-station data
# has been measured separately in the four development-plan sub-zones.

st.set_page_config(page_title="Ladywood Environmental Monitoring Dashboard",
                   layout="wide")
st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham, UK | Official council spatial layers")

CRVA_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)
BOUNDARIES_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Boundaries/MapServer"
)
BROWNFIELD_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer"
)

REQUEST_TIMEOUT = 60


def get_json(url, params=None):
    """Request JSON from an ArcGIS REST endpoint and raise readable errors."""
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        message = error.get("message", "ArcGIS REST request failed")
        details = error.get("details", [])
        if details:
            message += " — " + "; ".join(str(x) for x in details)
        raise RuntimeError(message)
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def get_ladywood_boundary():
    """Get Ladywood ward from the dedicated ward-boundary service."""
    url = f"{BOUNDARIES_URL}/0/query"
    params = {
        "where": "WARDNME = 'Ladywood'",
        "outFields": "WARDNME",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    data = get_json(url, params)
    features = data.get("features", [])
    records = []
    for feature in features:
        geom = feature.get("geometry")
        if geom:
            records.append({
                "WARDNME": feature.get("properties", {}).get("WARDNME", "Ladywood"),
                "geometry": shape(geom),
            })
    if not records:
        raise RuntimeError(
            "The ward service returned no Ladywood boundary. "
            "Check the service URL, field name, or network access."
        )
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf.dissolve(by="WARDNME", as_index=False)


@st.cache_data(ttl=3600, show_spinner=False)
def get_lsoa_scores(boundary_geojson):
    """Download the council's pre-calculated CRVA-by-LSOA layer, spatially filtered."""
    # The official CRVA-by-2021-LSOA layer is used as published.
    # No substitute score or invented factor values are calculated here.
    url = f"{CRVA_URL}/15/query"
    boundary = shape(boundary_geojson)
    minx, miny, maxx, maxy = boundary.bounds
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    data = get_json(url, params)
    rows = []
    for feature in data.get("features", []):
        geom = feature.get("geometry")
        if geom:
            row = feature.get("properties", {}).copy()
            row["geometry"] = shape(geom)
            rows.append(row)
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    # Clip to the Ladywood ward; keep only geometries that intersect it.
    gdf = gdf[gdf.geometry.intersects(boundary)].copy()
    gdf["geometry"] = gdf.geometry.intersection(boundary)
    gdf = gdf[~gdf.geometry.is_empty]
    return gdf


def arcgis_export_overlay(layer_id, bounds, opacity=0.65):
    """ Fetch one official MapServer layer as a transparent map image and place it over the ward. This also supports map-service layers that are not queryable as ordinary vector features. """
    minx, miny, maxx, maxy = bounds
    export_url = f"{CRVA_URL}/export"
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": "1200,900",
        "format": "png32",
        "transparent": "true",
        "layers": f"show:{layer_id}",
        "f": "image",
    }
    response = requests.get(export_url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        # ArcGIS may return a JSON error even when f=image was requested.
        try:
            payload = response.json()
            if payload.get("error"):
                raise RuntimeError(payload["error"].get("message", "Map export failed"))
        except ValueError:
            pass
        raise RuntimeError(f"Layer {layer_id} did not return a map image.")
    encoded = base64.b64encode(response.content).decode("ascii")
    image_url = f"data:image/png;base64,{encoded}"
    return ImageOverlay(
        image=image_url,
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=opacity,
        interactive=True,
        cross_origin=False,
        zindex=2,
    )


def add_legend(map_obj):
    legend_html = """ <div style=" position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white; border: 1px solid #777; border-radius: 4px; padding: 8px 10px; font-size: 12px; max-width: 240px;"> <b>Interpretation</b><br> CRVA layers are official council screening information.<br> Overlay colours follow the council's map symbology.<br> Missing data is not treated as zero. </div> """
    map_obj.get_root().html.add_child(folium.Element(legend_html))


# Factor names and official layer IDs from the council's CRVA_2025 MapServer.
# Layers 109 and 108 are the high- and medium-probability fluvial flood zones.
# Factors are displayed from the council map service; this app does not recreate
# factor scores where the official data is not provided as a queryable score.
LAYER_OPTIONS = {
    "Official CRVA score by 2021 LSOA": {"id": 15, "group": "Official CRVA"},
    "Fluvial flood — Zone 3 (high probability)": {"id": 109, "group": "Flooding"},
    "Fluvial flood — Zone 2 (medium probability)": {"id": 108, "group": "Flooding"},
    "Surface-water flooding": {"id": 1546, "group": "Flooding"},
    "Surface temperature (1981–2000 baseline)": {"id": 18, "group": "Climate"},
    "Local Climate Zones": {"id": 4, "group": "Climate"},
    "NO₂ concentration layer": {"id": 8, "group": "Air quality"},
    "PM₂.₅ concentration layer": {"id": 9, "group": "Air quality"},
    "Greenspace layer": {"id": 1549, "group": "Greenspace"},
    "Tree canopy / vegetation layer": {"id": 3, "group": "Greenspace"},
}

st.info(
    "The council's CRVA is a multi-factor screening assessment. The map layers "
    "below are shown as published by Birmingham City Council. The temperature "
    "layer uses a 1981–2000 baseline, so it must not be described as current "
    "temperature. Air-quality layers are not the same as measurements for each "
    "of the four development-plan sub-zones."
)

with st.sidebar:
    st.header("Map controls")
    selected_layers = st.multiselect(
        "Select official map layers",
        options=list(LAYER_OPTIONS.keys()),
        default=["Official CRVA score by 2021 LSOA",
                 "Surface-water flooding"],
    )
    opacity = st.slider("Overlay opacity", min_value=0.2, max_value=1.0,
                        value=0.65, step=0.05)
    show_boundary = st.checkbox("Show Ladywood ward boundary", value=True)
    show_layer_info = st.checkbox("Show layer notes", value=True)

try:
    boundary_gdf = get_ladywood_boundary()
    boundary_geom = boundary_gdf.geometry.unary_union
    boundary_bounds = boundary_geom.bounds
except Exception as exc:
    st.error(
        "Could not load the official Ladywood ward boundary. "
        "The dashboard stopped rather than drawing a guessed boundary."
    )
    st.code(str(exc))
    st.stop()

minx, miny, maxx, maxy = boundary_bounds
center = [(miny + maxy) / 2, (minx + maxx) / 2]
m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap",
               control_scale=True)

if show_boundary:
    folium.GeoJson(
        boundary_gdf.__geo_interface__,
        name="Ladywood ward boundary",
        style_function=lambda feature: {
            "color": "#d7191c",
            "weight": 3,
            "fill": False,
            "fillOpacity": 0,
        },
        tooltip=folium.GeoJsonTooltip(fields=["WARDNME"], aliases=["Ward:"]),
    ).add_to(m)

overlay_errors = []
for layer_name in selected_layers:
    layer_id = LAYER_OPTIONS[layer_name]["id"]
    try:
        overlay = arcgis_export_overlay(layer_id, boundary_bounds, opacity)
        overlay.layer_name = layer_name
        overlay.add_to(m)
    except Exception as exc:
        overlay_errors.append((layer_name, str(exc)))

folium.LayerControl(collapsed=False).add_to(m)
add_legend(m)
m.fit_bounds([[miny, minx], [maxy, maxx]])

left, right = st.columns([2.2, 1])
with left:
    st.subheader("Interactive environmental map")
    st_folium(m, use_container_width=True, height=650, returned_objects=[])

with right:
    st.subheader("Selected layers")
    if selected_layers:
        for name in selected_layers:
            item = LAYER_OPTIONS[name]
            st.markdown(f"**{name}**")
            st.caption(f"Official CRVA MapServer layer ID: {item['id']} · {item['group']}")
    else:
        st.write("Choose one or more layers from the sidebar.")

if overlay_errors:
    st.warning(
        "Some selected layers could not be rendered. Other layers remain available. "
        "The app does not replace failed layers with fabricated data."
    )
    for layer_name, message in overlay_errors:
        with st.expander(f"Error: {layer_name}"):
            st.code(message)

if show_layer_info:
    st.subheader("What this dashboard can and cannot claim")
    st.markdown(
        """ - **Official CRVA score:** use the council's published LSOA layer; do not label a project-created score as the official CRVA score. - **Flooding, climate, greenspace and vegetation layers:** these are spatial screening layers from the council service. Their displayed colours/symbology are supplied by the service. - **Air quality:** the NO₂ and PM₂.₅ map layers do not automatically provide measured monthly concentrations for each project sub-zone. A background monitoring station cannot be treated as four separate measurements without a defensible spatial model. - **Greenspace deficit:** a greenspace polygon layer is not, by itself, the council's calculated open-greenspace-deficit score. - **Brownfield land:** it is not one of the 11 CRVA factors. Add it as a separate indicator only after verifying the correct official brownfield register service and layer. - **Missing values:** missing data remain missing; they are never converted to zero. """
    )

st.caption(
    "Sources: Birmingham City Council CRVA 2025 MapServer and dedicated ward "
    "boundary service. Verify the service's layer metadata and methodology before "
    "using results for formal decisions."
)
