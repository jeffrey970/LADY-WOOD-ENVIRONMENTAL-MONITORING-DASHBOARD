import streamlit as st
import pandas as pd
import json
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping

st.set_page_config(page_title="Ladywood Environmental Monitoring Dashboard", layout="wide")
st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham | CRVA factors 3–9 | Official council data where published")

CRVA = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"
WARD_URL = f"{CRVA}/12/query"
TIMEOUT = 90

def arcgis_json(url, params):
    r = requests.get(url, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        err = data["error"]
        raise RuntimeError(err.get("message", "ArcGIS request failed") + " " +
                           "; ".join(err.get("details", [])))
    return data

@st.cache_data(ttl=3600, show_spinner=False)
def load_ladywood():
    """ Get Ladywood geometry from the CRVA service's dedicated Ward Boundaries layer (14), not the CRVA-by-Ward score layer (12). """
    url = f"{CRVA}/14/query"
    data = arcgis_json(url, {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "27700",
        "f": "geojson",
    })
    features = data.get("features", [])
    matched = []
    ward_field_candidates = ("WARDNME", "WARD_NAME", "NAME", "WardName", "WARD")
    for feature in features:
        props = feature.get("properties") or {}
        name = ""
        for field in ward_field_candidates:
            if field in props and props[field] is not None:
                name = str(props[field]).strip()
                if name:
                    break
        if name.casefold() == "ladywood" and feature.get("geometry"):
            matched.append({**props, "geometry": shape(feature["geometry"])})
    if not matched:
        sample = [
            {k: v for k, v in (f.get("properties") or {}).items()
             if "ward" in k.lower() or "name" in k.lower()}
            for f in features[:3]
        ]
        raise RuntimeError(
            "The dedicated CRVA ward-boundary layer (14) did not return a "
            "feature named Ladywood. "
            f"Returned {len(features)} features. Sample name fields: {sample}"
        )
    return gpd.GeoDataFrame(matched, geometry="geometry", crs="EPSG:27700")


@st.cache_data(ttl=3600, show_spinner=False)
def load_vector_layer(layer_id, boundary_geojson):
    boundary = shape(boundary_geojson)
    minx, miny, maxx, maxy = boundary.bounds
    url = f"{CRVA}/{layer_id}/query"
    # Return all intersecting features in the ward envelope, then clip exactly to ward.
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "27700", "outFields": "*", "returnGeometry": "true",
        "outSR": "27700", "f": "geojson"
    }
    data = arcgis_json(url, params)
    rows = []
    for f in data.get("features", []):
        if f.get("geometry"):
            rows.append({**(f.get("properties") or {}),
                         "geometry": shape(f["geometry"])})
    if not rows:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")
    gdf = gdf[gdf.geometry.intersects(boundary)].copy()
    gdf["geometry"] = gdf.geometry.intersection(boundary)
    return gdf[~gdf.geometry.is_empty].copy()

@st.cache_data(ttl=3600, show_spinner=False)
def load_raster_histogram(layer_id, boundary_geojson):
    """ Ask ArcGIS for raster VALUE/COUNT classes intersecting Ladywood. Some ArcGIS raster layers do not support this query; callers show that honestly. """
    boundary = shape(boundary_geojson)
    params = {
        "where": "1=1",
        "geometry": json.dumps(mapping(boundary)),
        "geometryType": "esriGeometryPolygon",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "27700",
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json"
    }
    data = arcgis_json(f"{CRVA}/{layer_id}/query", params)
    rows = data.get("features", [])
    vals = []
    for row in rows:
        attrs = row.get("attributes", {})
        value = attrs.get("VALUE", attrs.get("Value", attrs.get("value")))
        count = attrs.get("COUNT", attrs.get("Count", attrs.get("count")))
        if value is not None and count is not None:
            vals.append({"Class/value": str(value), "Pixel/cell count": float(count)})
    return pd.DataFrame(vals)

def chart_bar(df, x, y, title, xlabel=None, ylabel=None, rotation=35):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.tick_params(axis="x", labelrotation=rotation)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def query_coverage(layer_id, boundary):
    gdf = load_vector_layer(layer_id, mapping(boundary))
    if gdf.empty:
        return gdf, 0.0
    clipped_union = gdf.geometry.unary_union
    covered = clipped_union.intersection(boundary).area
    pct = min(100.0, 100.0 * covered / boundary.area) if boundary.area else 0.0
    return gdf, pct

with st.sidebar:
    st.header("Controls")
    show_map = st.checkbox("Show map", True)
    show_raw = st.checkbox("Show extracted tables", True)

try:
    ward = load_ladywood()
    boundary = ward.geometry.unary_union
except Exception as e:
    st.error("Could not retrieve Ladywood from the official CRVA ward layer.")
    st.code(str(e))
    st.stop()

boundary = boundary.buffer(0)
boundary_gdf = gpd.GeoDataFrame({"name": ["Ladywood"]}, geometry=[boundary], crs="EPSG:27700")
boundary_4326 = boundary_gdf.to_crs("EPSG:4326")
ward_row = ward.iloc[0]

st.subheader("Official CRVA ward summary")
cols = st.columns(4)
for col, label, field in zip(cols, ["Mean CRVA score", "Median CRVA score", "Minimum score", "Maximum score"],
                              ["MEAN", "MEDIAN", "MIN", "MAX"]):
    val = ward_row.get(field)
    col.metric(label, f"{float(val):.2f}" if pd.notna(val) else "Not available")
st.caption("These are the council's published combined ward-level CRVA statistics, not individual scores for factors 3–9.")

if show_map:
    geom = boundary_4326.geometry.iloc[0]
    bounds = geom.bounds
    m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2],
                   zoom_start=13, tiles="OpenStreetMap", control_scale=True)
    folium.GeoJson(boundary_4326.__geo_interface__,
                   name="Ladywood boundary",
                   style_function=lambda _: {"color": "red", "weight": 3, "fillOpacity": 0},
                   tooltip="Ladywood ward").add_to(m)
    folium.LayerControl().add_to(m)
    st.subheader("Ladywood boundary map")
    st_folium(m, use_container_width=True, height=450, returned_objects=[])

st.divider()
st.header("Factors 3–9: extracted data and charts")
st.write("The dashboard queries the council service and charts returned values. If a raster service does not expose a spatially filtered VALUE/COUNT table, the app reports that limitation instead of inventing values.")

# Factors 3 and 4: fluvial flood layers and surface-water flood extent are polygon features.
factor_results = []
vector_specs = [
    (3, "Fluvial flooding — Zone 3 high probability", 109),
    (3, "Fluvial flooding — Zone 2 medium probability", 108),
    (4, "Surface-water flooding extent", 1546),
]
for factor, name, layer_id in vector_specs:
    with st.spinner(f"Extracting {name}..."):
        try:
            layer, pct = query_coverage(layer_id, boundary)
            factor_results.append({
                "Factor": f"Factor {factor}",
                "Indicator": name,
                "Features returned": len(layer),
                "Ward area intersected (%)": round(pct, 2),
                "Status": "Extracted from official polygon layer"
            })
            with st.expander(f"{name} — extracted records"):
                st.write(f"Features intersecting Ladywood: **{len(layer)}**")
                st.write(f"Share of ward area covered by the union of features: **{pct:.2f}%**")
                st.caption("Coverage is calculated from clipped source polygons; it is not the official CRVA 0–1 factor score.")
                if show_raw and not layer.empty:
                    cols_to_show = [c for c in layer.columns if c != "geometry"]
                    st.dataframe(layer[cols_to_show].head(100), use_container_width=True)
        except Exception as e:
            factor_results.append({"Factor": f"Factor {factor}", "Indicator": name,
                                   "Features returned": None, "Ward area intersected (%)": None,
                                   "Status": f"Query failed: {e}"})

# Chart vector coverage where numeric values were actually extracted.
vector_df = pd.DataFrame(factor_results)
if not vector_df.empty:
    st.subheader("Flooding: mapped footprint within Ladywood")
    chart_df = vector_df.dropna(subset=["Ward area intersected (%)"]).copy()
    if not chart_df.empty:
        chart_bar(chart_df, "Indicator", "Ward area intersected (%)",
                  "Mapped flood polygon coverage within Ladywood",
                  ylabel="Share of ward area (%)", rotation=25)
    if show_raw:
        st.dataframe(vector_df, use_container_width=True)

# Factors 5, 6, 9 are raster layers; query their class/value counts if the REST service supports it.
raster_specs = [
    (5, "Surface temperature maximum (1981–2000)", 18),
    (6, "Local Climate Zones", 4),
    (9, "Tree canopy / vegetation raster", 3),
]
raster_tables = []
for factor, name, layer_id in raster_specs:
    with st.spinner(f"Requesting raster values for {name}..."):
        try:
            df = load_raster_histogram(layer_id, mapping(boundary))
            if df.empty:
                st.warning(f"{name}: the service did not return a spatially filtered value/count table. No chart was fabricated.")
                raster_tables.append({"Factor": factor, "Indicator": name, "Status": "No extractable ward-specific VALUE/COUNT table"})
            else:
                df["Indicator"] = name
                df["Factor"] = factor
                raster_tables.append({"Factor": factor, "Indicator": name, "Status": f"Extracted {len(df)} value classes"})
                st.subheader(name)
                chart_bar(df, "Class/value", "Pixel/cell count",
                          f"{name}: returned raster value counts",
                          ylabel="Cell/pixel count", rotation=0)
                if show_raw:
                    st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"{name}: raster values could not be extracted through the available query endpoint. {e}")
            raster_tables.append({"Factor": factor, "Indicator": name, "Status": f"Not extractable by this query: {e}"})

st.subheader("Factors 7–8: greenspace deficit")
st.warning(
    "The published CRVA service exposes a greenspace polygon layer (1549), buildings (1550), "
    "and woodland (1552), but does not expose separate, pre-calculated Ladywood tables for "
    "the official open-greenspace-deficit and other-greenspace-deficit scores. A polygon count "
    "or greenspace area is not the same as the council's 100 m grid deficit score. These two "
    "official scores are therefore not plotted as if they had been extracted."
)
try:
    greenspace = load_vector_layer(1549, mapping(boundary))
    if not greenspace.empty:
        greenspace["clipped_area_m2"] = greenspace.geometry.area
        by_type = (greenspace.groupby("type" if "type" in greenspace.columns else greenspace.columns[0], dropna=False)["clipped_area_m2"]
                   .sum().reset_index().sort_values("clipped_area_m2", ascending=False))
        st.subheader("Greenspace source data available in Ladywood (supporting evidence, not deficit score)")
        chart_bar(by_type, "type", "clipped_area_m2",
                  "Area of mapped greenspace polygons by source category",
                  ylabel="Area (m²)", rotation=35)
        if show_raw:
            st.dataframe(by_type, use_container_width=True)
except Exception as e:
    st.warning(f"Could not extract greenspace source polygons: {e}")

st.subheader("Extraction status")
status_df = pd.DataFrame(raster_tables)
if not status_df.empty:
    st.dataframe(status_df, use_container_width=True)
st.markdown(
    """ **Interpretation rules** - Flood polygon coverage is a project-calculated spatial summary, not an official CRVA factor score. - Temperature raster is a historical 1981–2000 baseline, not a current temperature reading. - Local climate zone values are categorical classes; they should not be averaged as if they were temperatures. - Tree canopy raster query failures are reported, not replaced with guessed canopy percentages. - The official CRVA methodology uses a 100 m grid and standardised factor scores. The ward-level combined score above is supplied by the council. """
)
st.markdown(
    f"**Sources:** [Birmingham City Council CRVA map]({CRVA}) · "
    "[Council CRVA methodology](https://research.birmingham.ac.uk/files/253369474/CRVA_methodology_guidancedoc_v1.1.pdf)"
)
