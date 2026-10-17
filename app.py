Environmental_Assessment.csv", mime="text/csv") import streamlit as st
import pandas as pd
import requests
import folium
from shapely.geometry import shape, Polygon, MultiPolygon, LineString, MultiLineString, Point, box, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer

# ============================================================
# PAGE AND SOURCES
# ============================================================
st.set_page_config(page_title="Ladywood Environmental Monitoring Dashboard", page_icon="🌍", layout="wide")
st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham, UK Environmental Dashboard")

CRVA = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"
WARD_LAYER = CRVA + "/14"
LSOA_LAYER = CRVA + "/17"
SURFACE_FLOOD_LAYER = CRVA + "/1546"
BROWNFIELD_LAYER = "https://maps.birmingham.gov.uk/server/rest/services/planning/HELAA/MapServer/38"
ROAD_LAYER = "https://maps.birmingham.gov.uk/server/rest/services/Internet_Highways_open/MapServer/26"

BNG, WGS84 = "EPSG:27700", "EPSG:4326"
to_wgs84 = Transformer.from_crs(BNG, WGS84, always_xy=True).transform
session = requests.Session()

# ============================================================
# ARCGIS HELPERS
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
                    p = Polygon(ring)
                    if not p.is_valid:
                        p = p.buffer(0)
                    if not p.is_empty:
                        polygons.append(p)
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
                    pass
        if not lines:
            return None
        return lines[0] if len(lines) == 1 else MultiLineString(lines)
    if "x" in geom and "y" in geom:
        return Point(geom["x"], geom["y"])
    return None


def arcgis_request(layer_url, where="1=1", bbox=None, offset=None, ids_only=False):
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
        # Use a proper ArcGIS envelope JSON object rather than a bare comma string.
        xmin, ymin, xmax, ymax = bbox
        params["geometry"] = (
            '{"xmin":%s,"ymin":%s,"xmax":%s,"ymax":%s,"spatialReference":{"wkid":27700}}'
            % (xmin, ymin, xmax, ymax)
        )
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"
    if offset is not None:
        params["resultOffset"] = offset
        params["resultRecordCount"] = 1000
    response = session.get(layer_url + "/query", params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise RuntimeError(f"ArcGIS error for {layer_url}: {data['error']}")
    return data


def get_features(layer_url, where="1=1", bbox=None):
    """Fetch all features, using object IDs when available to avoid paging gaps."""
    first = arcgis_request(layer_url, where=where, bbox=bbox)
    features = first.get("features", [])
    if first.get("exceededTransferLimit", False):
        offset = len(features)
        while True:
            page = arcgis_request(layer_url, where=where, bbox=bbox, offset=offset)
            next_features = page.get("features", [])
            features.extend(next_features)
            if not page.get("exceededTransferLimit", False) or not next_features:
                break
            offset += len(next_features)
    out = []
    for feature in features:
        row = dict(feature.get("attributes", {}))
        row["geometry"] = esri_geometry_to_shape(feature.get("geometry"))
        out.append(row)
    return out


def load_spatial_layer(layer_url, label, ward_geom, bbox):
    """Load polygons intersecting Ladywood; report failures instead of silently treating them as zero exposure."""
    try:
        rows = get_features(layer_url, bbox=bbox)
        feats = []
        for row in rows:
            geom = row.get("geometry")
            if geom is None or geom.is_empty or not geom.intersects(ward_geom):
                continue
            clipped = geom.intersection(ward_geom)
            if not clipped.is_empty:
                feats.append({"attributes": row, "geometry": clipped})
        return feats, None
    except Exception as exc:
        st.warning(f"Could not load {label}: {exc}")
        return [], str(exc)


def exposure_percent(area_geom, features, data_loaded=True):
    # None means the layer failed to load; zero means it loaded but no overlap was found.
    if not data_loaded:
        return None
    if not features:
        return 0.0
    intersections = []
    for feat in features:
        geom = feat.get("geometry")
        if geom is not None and not geom.is_empty and geom.intersects(area_geom):
            inter = geom.intersection(area_geom)
            if not inter.is_empty:
                intersections.append(inter)
    if not intersections or area_geom.area <= 0:
        return 0.0
    return 100 * unary_union(intersections).area / area_geom.area


def normalise(series):
    numeric = pd.to_numeric(series, errors="coerce")
    low, high = numeric.min(), numeric.max()
    if pd.isna(low) or pd.isna(high) or high == low:
        return pd.Series([0.0 if pd.notna(x) else float("nan") for x in numeric], index=series.index)
    return (numeric - low) / (high - low)


def risk_colour(value):
    return {"HIGH": "red", "MODERATE": "orange", "LOW": "green"}.get(value, "gray")

# ============================================================
# OFFICIAL LADYWOOD WARD
# ============================================================
try:
    ward_rows = get_features(WARD_LAYER, where="WARDNME = 'Ladywood'")
except Exception:
    ward_rows = []
if not ward_rows:
    try:
        ward_rows = [r for r in get_features(WARD_LAYER) if str(r.get("WARDNME", "")).strip().lower() == "ladywood"]
    except Exception as exc:
        st.error(f"Could not retrieve Birmingham ward boundaries: {exc}")
        st.stop()
if not ward_rows or ward_rows[0].get("geometry") is None:
    st.error("The official Ladywood ward boundary could not be retrieved.")
    st.stop()
ladywood_geometry = ward_rows[0]["geometry"].buffer(0)
if ladywood_geometry.is_empty:
    st.error("The official Ladywood ward geometry is empty.")
    st.stop()
ladywood_bbox = ladywood_geometry.bounds

# ============================================================
# OFFICIAL LSOAs CLIPPED TO LADYWOOD
# ============================================================
try:
    lsoa_rows = get_features(LSOA_LAYER, bbox=ladywood_bbox)
except Exception as exc:
    st.error(f"The official LSOA dataset could not be loaded: {exc}")
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
        "LSOA21CD": row.get("LSOA21CD"), "LSOA21NM": row.get("LSOA21NM"),
        "MEAN": row.get("MEAN"), "AVERAGE_RISK": row.get("AVERAGE_RISK"),
        "MINIMUM_RISK": row.get("MINIMUM_RISK"), "MAXIMUM_RISK": row.get("MAXIMUM_RISK"),
        "geometry": clipped, "Area_m2": clipped.area,
    })
lsoa_df = pd.DataFrame(lsoa_records)
if lsoa_df.empty:
    st.error("No official LSOA areas intersecting Ladywood were found.")
    st.stop()
lsoa_df["CRVA_MEAN"] = pd.to_numeric(lsoa_df["MEAN"], errors="coerce")

# ============================================================
# LOAD ENVIRONMENTAL DATA
# ============================================================
surface_features, surface_error = load_spatial_layer(SURFACE_FLOOD_LAYER, "Surface flooding", ladywood_geometry, ladywood_bbox)
brownfield_features, brownfield_error = load_spatial_layer(BROWNFIELD_LAYER, "Brownfield register", ladywood_geometry, ladywood_bbox)
road_features, road_error = load_spatial_layer(ROAD_LAYER, "Classified roads", ladywood_geometry, ladywood_bbox)

# ============================================================
# CALCULATE EXPOSURE BY LSOA
# ============================================================
results = []
for _, row in lsoa_df.iterrows():
    geom = row["geometry"]
    surface = exposure_percent(geom, surface_features, data_loaded=(surface_error is None))
    brown = exposure_percent(geom, brownfield_features, data_loaded=(brownfield_error is None))
    results.append({
        "LSOA21CD": row["LSOA21CD"], "LSOA21NM": row["LSOA21NM"], "Area_m2": row["Area_m2"],
        "CRVA_MEAN": row["CRVA_MEAN"], "CRVA_AVERAGE_RISK": row["AVERAGE_RISK"],
        "Surface_Flood_%": surface,
        "Brownfield_Exposure_%": brown,
        "geometry": geom,
    })
risk_df = pd.DataFrame(results)
risk_df["Climate_Index"] = normalise(risk_df["CRVA_MEAN"])
risk_df["Surface_Flood_Index"] = normalise(risk_df["Surface_Flood_%"])
risk_df["Brownfield_Index"] = normalise(risk_df["Brownfield_Exposure_%"])
# The equal-weight index and cutoffs are project-defined screening choices, not official statutory risk ratings. This must be disclosed in the report.
valid_components = risk_df[["Climate_Index", "Surface_Flood_Index", "Brownfield_Index"]]
risk_df["Environmental_Index"] = valid_components.mean(axis=1, skipna=False)

def classify_risk(value):
    if pd.isna(value):
        return "NOT AVAILABLE"
    if value >= 0.67:
        return "HIGH"
    if value >= 0.34:
        return "MODERATE"
    return "LOW"
risk_df["Risk_Class"] = risk_df["Environmental_Index"].apply(classify_risk)

def dominant_driver(row):
    vals = {"Climate vulnerability": row["Climate_Index"], "Surface flooding": row["Surface_Flood_Index"], "Brownfield exposure": row["Brownfield_Index"]}
    vals = {k: v for k, v in vals.items() if pd.notna(v)}
    return max(vals, key=vals.get) if vals else "Not available"
risk_df["Dominant_Environmental_Driver"] = risk_df.apply(dominant_driver, axis=1)

# ============================================================
# NEAREST CLASSIFIED ROAD
# ============================================================
def nearest_road_info(area_geom, roads):
    nearest_distance, nearest_class = None, None
    for road in roads:
        geom = road.get("geometry")
        if geom is None or geom.is_empty:
            continue
        distance = area_geom.distance(geom)
        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_class = road.get("attributes", {}).get("BRUM_CLASS")
    return nearest_distance, nearest_class
road_distances = [nearest_road_info(row["geometry"], road_features) for _, row in risk_df.iterrows()]
risk_df["Nearest_Classified_Road_m"] = [x[0] for x in road_distances]
risk_df["Nearest_Road_Class"] = [x[1] for x in road_distances]

# ============================================================
# MAP BASE AND BOUNDARY
# ============================================================
map_center_wgs84 = transform(to_wgs84, ladywood_geometry.centroid)
m = folium.Map(location=[map_center_wgs84.y, map_center_wgs84.x], zoom_start=14, control_scale=True, tiles="OpenStreetMap")
ladywood_wgs = transform(to_wgs84, ladywood_geometry)
folium.GeoJson(mapping(ladywood_wgs), name="Ladywood Boundary", style_function=lambda f: {"color": "black", "weight": 3, "fillOpacity": 0}, tooltip="Official Ladywood Ward Boundary").add_to(m)
legend_html = '''<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:12px;border:1px solid #777;font-size:13px;box-shadow:0 1px 5px rgba(0,0,0,.3)"><b>Environmental Screening</b><br><span style="color:green">●</span> Low<br><span style="color:orange">●</span> Moderate<br><span style="color:red">●</span> High<br><span style="color:grey">●</span> Not available</div>'''
m.get_root().html.add_child(folium.Element(legend_html))

# LSOA screening polygons
lsoa_group = folium.FeatureGroup(name="Ladywood LSOA Risk", show=True)
for _, row in risk_df.iterrows():
    geom_wgs = transform(to_wgs84, row["geometry"])
    colour = risk_colour(row["Risk_Class"])
    env_idx = row["Environmental_Index"]
    flood_val = row["Surface_Flood_%"]
    popup = (f"<b>{row['LSOA21NM']}</b><br>LSOA: {row['LSOA21CD']}<br>Screening class: {row['Risk_Class']}<br>Environmental index: {env_idx:.2f}" if pd.notna(env_idx) else f"<b>{row['LSOA21NM']}</b><br>Screening class: {row['Risk_Class']}<br>Environmental index: Not available")
    popup += f"<br>Climate vulnerability index: {row['Climate_Index']:.2f}" if pd.notna(row["Climate_Index"]) else "<br>Climate vulnerability index: Not available"
    popup += f"<br>Surface flooding exposure: {flood_val:.1f}%" if pd.notna(flood_val) else "<br>Surface flooding exposure: Not available"
    popup += f"<br>Brownfield exposure: {row['Brownfield_Exposure_%']:.1f}%" if pd.notna(row["Brownfield_Exposure_%"]) else "<br>Brownfield exposure: Not available"
    popup += f"<br>Dominant driver: {row['Dominant_Environmental_Driver']}"
    dist = row["Nearest_Classified_Road_m"]
    popup += f"<br>Nearest classified road: {dist:.0f} m" if pd.notna(dist) else "<br>Nearest classified road: Not available"
    folium.GeoJson(mapping(geom_wgs), style_function=lambda f, c=colour: {"color": c, "weight": 2, "fillColor": c, "fillOpacity": 0.20}, popup=folium.Popup(popup, max_width=400)).add_to(lsoa_group)
lsoa_group.add_to(m)

# Surface flooding
surface_group = folium.FeatureGroup(name="Surface Flooding", show=False)
for feat in surface_features:
    geom_wgs = transform(to_wgs84, feat["geometry"])
    folium.GeoJson(mapping(geom_wgs), style_function=lambda f: {"color": "purple", "weight": 1.5, "fillColor": "purple", "fillOpacity": 0.25}, tooltip="Surface Flooding").add_to(surface_group)
surface_group.add_to(m)

# Brownfield register
brown_group = folium.FeatureGroup(name="Brownfield Register", show=False)
for feat in brownfield_features:
    geom_wgs = transform(to_wgs84, feat["geometry"])
    folium.GeoJson(mapping(geom_wgs), style_function=lambda f: {"color": "#7B3F00", "weight": 2, "fillColor": "#A0522D", "fillOpacity": 0.30}, tooltip="Brownfield Register").add_to(brown_group)
brown_group.add_to(m)

# Classified roads
road_group = folium.FeatureGroup(name="Classified Roads", show=False)
def road_colour(road_class):
    if road_class == "A Road": return "red"
    if road_class == "B Road": return "blue"
    return "gold"
for feat in road_features:
    geom = feat.get("geometry")
    if geom is None or geom.is_empty: continue
    geom_wgs = transform(to_wgs84, geom)
    attrs = feat.get("attributes", {})
    road_class = attrs.get("BRUM_CLASS", "Classified Road")
    road_name = attrs.get("BRUM_ROAD_", "")
    folium.GeoJson(mapping(geom_wgs), style_function=lambda f, rc=road_class: {"color": road_colour(rc), "weight": 3}, tooltip=f"{road_class}" + (f" — {road_name}" if road_name else "")).add_to(road_group)
road_group.add_to(m)

# Risk markers
marker_group = folium.FeatureGroup(name="Risk Markers", show=True)
for _, row in risk_df.iterrows():
    p = transform(to_wgs84, row["geometry"].centroid)
    colour = risk_colour(row["Risk_Class"])
    folium.CircleMarker(location=[p.y, p.x], radius=7, color=colour, fill=True, fill_color=colour, fill_opacity=0.9, tooltip=f"{row['LSOA21NM']} — {row['Risk_Class']}", popup=f"<b>{row['LSOA21NM']}</b><br>Risk: {row['Risk_Class']}<br>Main driver: {row['Dominant_Environmental_Driver']}").add_to(marker_group)
marker_group.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

st.subheader("Ladywood Environmental Map")
st.caption("Use the layer control in the top-right of the map to turn environmental layers on and off.")
st.components.v1.html(m.get_root().render(), height=680, scrolling=False)

# ============================================================
# TABLES AND CHARTS
# ============================================================
st.subheader("Environmental Assessment Classification")
st.caption("This table compares official Ladywood LSOA areas using climate vulnerability, flood exposure and brownfield exposure.")
table_columns = ["LSOA21CD", "LSOA21NM", "CRVA_MEAN", "CRVA_AVERAGE_RISK", "Surface_Flood_%", "Brownfield_Exposure_%", "Environmental_Index", "Risk_Class", "Dominant_Environmental_Driver"]
display_df = risk_df[table_columns].sort_values("Environmental_Index", ascending=False, na_position="last")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.subheader("Priority Ladywood Areas for Further Investigation")
priority_df = risk_df[risk_df["Risk_Class"] == "HIGH"].sort_values("Environmental_Index", ascending=False)
if priority_df.empty:
    st.info("No Ladywood LSOA reached the HIGH assessment category using this project screening index.")
    moderate_df = risk_df[risk_df["Risk_Class"] == "MODERATE"].sort_values("Environmental_Index", ascending=False)
    if not moderate_df.empty:
        st.write("**Highest MODERATE assessment areas:**")
        st.dataframe(moderate_df[["LSOA21CD", "LSOA21NM", "Environmental_Index", "Dominant_Environmental_Driver"]], use_container_width=True, hide_index=True)
else:
    st.dataframe(priority_df[["LSOA21CD", "LSOA21NM", "Environmental_Index", "Dominant_Environmental_Driver"]], use_container_width=True, hide_index=True)

st.subheader("Environmental Risk by Ladywood Area")
st.caption("Higher values indicate higher combined project assessment scores")
st.bar_chart(risk_df.set_index("LSOA21NM")[["Environmental_Index"]])
st.subheader("Climate Vulnerability by Ladywood Area")
st.caption("Official numeric MEAN value from Birmingham's 2025 LSOA environmental-risk dataset.")
st.bar_chart(risk_df.set_index("LSOA21NM")[["CRVA_MEAN"]])
st.subheader("Surface Flooding Exposure by Ladywood Area")
st.caption("Percentage of each LSOA intersecting the available surface-flooding extent. Missing data is not treated as zero.")
st.bar_chart(risk_df.set_index("LSOA21NM")[["Surface_Flood_%"]])
st.subheader("Brownfield Exposure by Ladywood Area")
st.caption("Percentage of each LSOA intersecting the Birmingham Brownfield Register layer.")
st.bar_chart(risk_df.set_index("LSOA21NM")[["Brownfield_Exposure_%"]])
st.subheader("Environmental Drivers by Ladywood Area")
st.caption("Normalised indicators allow the available environmental factors to be compared on the same 0–1 scale.")
st.bar_chart(risk_df.set_index("LSOA21NM")[["Climate_Index", "Surface_Flood_Index", "Brownfield_Index"]])

st.subheader("Classified Roads Within Ladywood")
st.caption("Official Birmingham classified-road data are shown as environmental context. Road proximity is reported separately rather than assigned an arbitrary pollution multiplier.")
road_records = [{"Road classification": f.get("attributes", {}).get("BRUM_CLASS"), "Road": f.get("attributes", {}).get("BRUM_ROAD_")} for f in road_features]
if road_records:
    roads_df = pd.DataFrame(road_records).drop_duplicates().sort_values(["Road classification", "Road"], na_position="last")
    st.dataframe(roads_df, use_container_width=True, hide_index=True)
else:
    st.warning("No classified-road features were returned inside Ladywood.")
st.subheader("Classified Road Proximity by Ladywood Area")
st.dataframe(risk_df[["LSOA21CD", "LSOA21NM", "Nearest_Classified_Road_m", "Nearest_Road_Class"]].sort_values("Nearest_Classified_Road_m", na_position="last"), use_container_width=True, hide_index=True)
st.subheader("Engineering Assessment Interpretation File")
st.caption("Indicators identify where further engineering investigation may be useful.")

# ============================================================
# DOWNLOAD
# ============================================================
download_columns = ["LSOA21CD", "LSOA21NM", "Area_m2", "CRVA_MEAN", "CRVA_AVERAGE_RISK", "Climate_Index", "Surface_Flood_%", "Surface_Flood_Index", "Brownfield_Exposure_%", "Brownfield_Index", "Environmental_Index", "Risk_Class", "Dominant_Environmental_Driver", "Nearest_Classified_Road_m", "Nearest_Road_Class"]
download_df = risk_df[download_columns].copy()
st.download_button(label="Download Ladywood Environmental Assessment Results", data=download_df.to_csv(index=False), file_name="Ladywood_
