""" Ladywood Environmental Investigation Dashboard Source: Birmingham City Council CRVA_2025 ArcGIS REST service. Climate/temperature/local-climate-zone layers and composite CRVA score are excluded. Install: pip install streamlit pandas requests plotly geopandas shapely pyproj streamlit-folium folium Run: streamlit run ladywood_dashboard.py This app downloads source layers live when the user clicks "Load council data". It preserves source attributes, logs failed layers, and does not turn missing data into zero. """
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"
OUT = Path("Ladywood_CRVA_Extract")
OUT.mkdir(exist_ok=True)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Ladywood-Environmental-Student-Prototype/1.0"})

st.set_page_config(page_title="Ladywood Environmental Investigation Dashboard", page_icon="🗺️", layout="wide")
st.title("Ladywood Environmental Investigation Dashboard")
st.caption("Birmingham, UK | Official CRVA 2025 layers | Climate layers excluded")

st.warning(
    "This is an investigation-screening prototype, not an official statutory risk rating. "
    "Relative pollution layers are not automatically measured concentrations. Missing data is not zero. "
    "Polygon layers returned by an intersects query may extend beyond the ward unless clipped in GIS."
)

FEATURE_LAYERS = {
    "LSOA boundaries": 17,
    "Greenspace": 1549,
    "Buildings": 1550,
    "Woodland": 1552,
    "Deprivation index (2019)": 1545,
    "Excess years of life lost": 7,
    "Fluvial flood zone 3 (high probability)": 109,
    "Fluvial flood zone 2 (medium probability)": 108,
    "Surface-water flooding extent": 1546,
}
RASTER_LAYERS = {
    "PM2.5 relative pollutant level": 9,
    "NO₂ relative pollutant level": 8,
    "Tree-canopy cover hillshade": 3,
}
# Climate layers deliberately omitted: temperature (18) and local climate zones (4).
# Composite CRVA score omitted to avoid inadvertently including climate components.

def arcgis_json(url, params=None):
    q = dict(params or {})
    q["f"] = "json"
    r = SESSION.get(url, params=q, timeout=90)
    r.raise_for_status()
    obj = r.json()
    if "error" in obj:
        raise RuntimeError(str(obj["error"]))
    return obj

def layer_metadata(layer_id):
    return arcgis_json(f"{BASE}/{layer_id}")

def query_layer(layer_id, where="1=1", geometry=None, geometry_type="esriGeometryPolygon", in_sr=27700, out_sr=4326, out_fields="*", return_geometry=True):
    url = f"{BASE}/{layer_id}/query"
    all_features, offset = [], 0
    while True:
        params = {
            "where": where, "outFields": out_fields,
            "returnGeometry": str(return_geometry).lower(),
            "outSR": out_sr, "resultRecordCount": 2000, "resultOffset": offset,
        }
        if geometry is not None:
            params.update({
                "geometry": json.dumps(geometry, separators=(",", ":")),
                "geometryType": geometry_type, "inSR": in_sr,
                "spatialRel": "esriSpatialRelIntersects",
            })
        obj = arcgis_json(url, params)
        feats = obj.get("features", [])
        all_features.extend(feats)
        if not obj.get("exceededTransferLimit") or not feats:
            break
        offset += len(feats)
    return all_features

def arc_geometry_to_geojson(g):
    if not g:
        return None
    if "rings" in g:
        # ArcGIS polygon rings can contain holes; this basic conversion preserves rings
        # but represents them as a Polygon. Inspect geometry before area calculations.
        return {"type": "Polygon", "coordinates": g["rings"]}
    if "paths" in g:
        return {"type": "MultiLineString", "coordinates": g["paths"]}
    if "x" in g and "y" in g:
        return {"type": "Point", "coordinates": [g["x"], g["y"]]}
    return None

def save_layer(name, features):
    (OUT / f"{name}.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": arc_geometry_to_geojson(f.get("geometry")),
            "properties": f.get("attributes", {}),
        } for f in features]
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([f.get("attributes", {}) for f in features]).to_csv(
        OUT / f"{name}.csv", index=False, encoding="utf-8-sig"
    )

def load_council_data():
    log, data, metadata = [], {}, {}
    ward_features = query_layer(14, where="UPPER(WARDNME) = 'LADYWOOD'",
                                out_sr=27700, out_fields="*")
    if not ward_features:
        ward_features = query_layer(14, where="WARDNME = 'Ladywood'",
                                    out_sr=27700, out_fields="*")
    if not ward_features:
        raise RuntimeError("Could not retrieve the official Ladywood ward boundary.")
    ward = ward_features[0]
    ward_geom = ward.get("geometry")
    if not ward_geom:
        raise RuntimeError("Ward query returned no geometry; extraction stopped safely.")
    save_layer("Ladywood_Ward_Boundary", ward_features)
    data["Official Ladywood ward boundary"] = pd.DataFrame(
        [f.get("attributes", {}) for f in ward_features]
    )
    log.append({"Layer": "Official Ladywood ward boundary", "ID": 14,
                "Records": len(ward_features), "Status": "OK"})

    # LSOA polygons are selected by spatial intersection with the official ward.
    for name, layer_id in FEATURE_LAYERS.items():
        try:
            metadata[name] = layer_metadata(layer_id)
            feats = query_layer(layer_id, geometry=ward_geom, out_sr=4326)
            save_layer(name.replace(" ", "_").replace("/", "_"), feats)
            data[name] = pd.DataFrame([f.get("attributes", {}) for f in feats])
            log.append({"Layer": name, "ID": layer_id, "Records": len(feats), "Status": "OK"})
        except Exception as e:
            data[name] = pd.DataFrame()
            log.append({"Layer": name, "ID": layer_id, "Records": None, "Status": f"FAILED: {e}"})

    # Raster layers are metadata-only in the dashboard. Export images are available via the REST service.
    for name, layer_id in RASTER_LAYERS.items():
        try:
            metadata[name] = layer_metadata(layer_id)
            log.append({"Layer": name, "ID": layer_id, "Records": "Raster", "Status": "Metadata OK; raster not converted to invented table"})
        except Exception as e:
            log.append({"Layer": name, "ID": layer_id, "Records": None, "Status": f"FAILED: {e}"})
    return data, metadata, pd.DataFrame(log)

def numeric_columns(df):
    if df.empty:
        return []
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]

def show_table_and_chart(label, df):
    st.subheader(label)
    if df.empty:
        st.info("No records returned for this layer. Check the extraction log; this is not evidence of zero risk.")
        return
    st.caption(f"{len(df):,} returned records · {len(df.columns)} source attributes")
    nums = numeric_columns(df)
    if nums:
        chosen = st.selectbox(f"Numeric attribute to inspect — {label}", nums, key=f"num_{label}")
        vals = pd.to_numeric(df[chosen], errors="coerce").dropna()
        if len(vals):
            left, mid, right = st.columns(3)
            left.metric("Valid values", f"{len(vals):,}")
            mid.metric("Minimum", f"{vals.min():,.3g}")
            right.metric("Maximum", f"{vals.max():,.3g}")
            # Plot source attribute values only; no fabricated geographic inference.
            plot_df = pd.DataFrame({"Source record": range(1, len(vals) + 1), chosen: vals.values})
            st.plotly_chart(px.histogram(plot_df, x=chosen, title=f"Distribution of source attribute: {chosen}"),
                            use_container_width=True)
    else:
        st.info("No numeric source attributes detected. This layer is suitable for spatial inspection, not a numeric chart as returned.")
    st.dataframe(df.head(1000), use_container_width=True)
    st.download_button(f"Download {label} CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{label.replace(' ', '_')}.csv", mime="text/csv",
                       key=f"dl_{label}")

with st.sidebar:
    st.header("Controls")
    st.write("Use the button to query the council service live.")
    reload_data = st.button("Load / refresh council data", type="primary", use_container_width=True)
    st.divider()
    st.write("Included source layers")
    for n, i in FEATURE_LAYERS.items():
        st.caption(f"{i} · {n}")
    for n, i in RASTER_LAYERS.items():
        st.caption(f"{i} · {n}")
    st.caption("Excluded: composite CRVA score, summer temperature, local climate zones.")

if "loaded" not in st.session_state:
    st.session_state.loaded = False
if reload_data:
    with st.spinner("Querying Birmingham City Council's ArcGIS service…"):
        try:
            d, m, log_df = load_council_data()
            st.session_state.data = d
            st.session_state.metadata = m
            st.session_state.log = log_df
            st.session_state.loaded = True
            st.session_state.load_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        except Exception as e:
            st.error(f"Extraction stopped: {e}")
            st.info("No risk calculations have been made. Check network access and the council service status.")

if st.session_state.loaded:
    data = st.session_state.data
    metadata = st.session_state.metadata
    log_df = st.session_state.log
    st.caption(f"Last load: {st.session_state.load_time}")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Individual layers", "Data quality", "Methods & limitations"])
    with tab1:
        st.subheader("Extraction overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Feature layers configured", len(FEATURE_LAYERS))
        c2.metric("Raster layers identified", len(RASTER_LAYERS))
        c3.metric("Feature records returned", sum(len(x) for x in data.values()))
        st.dataframe(log_df, use_container_width=True)
        st.subheader("Returned record counts")
        count_df = log_df.copy()
        count_df["Records numeric"] = pd.to_numeric(count_df["Records"], errors="coerce")
        chart_df = count_df.dropna(subset=["Records numeric"])
        if not chart_df.empty:
            st.plotly_chart(px.bar(chart_df, x="Layer", y="Records numeric",
                                   title="Records returned by source layer", color="Layer"),
                            use_container_width=True)
        st.download_button("Download extraction log", log_df.to_csv(index=False).encode("utf-8-sig"),
                           "extraction_log.csv", "text/csv")
    with tab2:
        choice = st.selectbox("Choose a source layer", list(data.keys()))
        show_table_and_chart(choice, data[choice])
        raster_choice = st.selectbox("Raster metadata", list(RASTER_LAYERS.keys()))
        st.json(metadata.get(raster_choice, {"message": "Metadata unavailable"}))
        st.caption("Raster layers are not charted as numeric values because an image export is not a validated cell-value table.")
    with tab3:
        st.subheader("Data quality checks")
        quality = []
        for name, df in data.items():
            quality.append({
                "Layer": name, "Rows": len(df), "Columns": len(df.columns),
                "Empty result": df.empty,
                "Duplicate rows": int(df.duplicated().sum()) if not df.empty else 0,
                "Missing cells": int(df.isna().sum().sum()) if not df.empty else 0,
            })
        st.dataframe(pd.DataFrame(quality), use_container_width=True)
        st.write("A returned polygon can intersect Ladywood without lying wholly inside it. Clip/intersect geometry in a GIS before calculating within-ward area percentages.")
        st.write("Do not average source codes or ID fields. Inspect layer metadata to identify the meaning, units, year and direction of every numeric field.")
    with tab4:
        st.markdown(""" **Purpose:** identify environmental evidence that warrants further investigation in Ladywood. **Source:** Birmingham City Council, CRVA 2025 ArcGIS REST service. **Excluded:** the composite CRVA assessment layer, summer temperature and local-climate-zone layers, because the request is to exclude climate. **Important limitations** - PM2.5 and NO₂ layers are described by the service as relative levels (lowest to highest). Do not label them as µg/m³ or compare them directly with WHO concentration thresholds unless the source metadata confirms units and calibration. - Raster layer metadata is retrieved, but raster cell values are not silently converted into a table. A validated raster download and zonal-statistics workflow is needed for LSOA-level numeric comparisons. - The layer query uses spatial intersection. It does not guarantee all returned features are clipped to the ward. - Deprivation and excess-years-of-life-lost are contextual vulnerability indicators, not direct measurements of pollution or flood hazard. - The dashboard deliberately does not calculate a combined risk score or rank areas until fields, units, directionality, geometry and missingness have been verified. - Empty/failed data is shown as unavailable, never as zero. **Validation before presenting:** compare record counts with the service, inspect source attributes and metadata, verify spatial reference and geometry, then document each derived indicator and test it against known examples. """)
else:
    st.info("Click **Load / refresh council data** in the sidebar to retrieve the available non-climate layers.")
    st.markdown("### What this prototype is designed to show")
    st.markdown(""" - Individual source-layer tables and numeric attribute distributions. - A retrieval log with layer IDs, record counts and failures. - Separate raster metadata for PM2.5, NO₂ and tree canopy. - Downloadable CSV/GeoJSON files for subsequent GIS validation. - Explicit limitations so the dashboard does not overstate what the data proves. """)
