import streamlit as st
import pandas as pd
import requests
import folium
import streamlit.components.v1 as components

from shapely.geometry import shape
from shapely.ops import unary_union
from pyproj import Transformer

# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Dashboard")
st.caption("Environmental screening of Ladywood using official spatial data.")

# ---------------------------------------------------------
# OFFICIAL DATA SOURCES
# ---------------------------------------------------------

CRVA = "https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer"

WARD_URL = CRVA + "/14"
LSOA_URL = CRVA + "/17"
FLOOD_URL = CRVA + "/109"
SURFACE_FLOOD_URL = CRVA + "/1546"

BROWNFIELD_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROADS_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer/25"
)

AIR_PAGE = "https://uk-air.defra.gov.uk/data/flat_files?site_id=BMLD"

# ---------------------------------------------------------
# GET ARCGIS DATA
# ---------------------------------------------------------

def get_data(url):
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson"
    }

    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()

    data = r.json()

    if "features" not in data:
        raise ValueError("No spatial features returned.")

    return data


def features_to_shapes(data):
    result = []

    for feature in data["features"]:
        if feature.get("geometry"):
            try:
                geom = shape(feature["geometry"])
                result.append((feature, geom))
            except Exception:
                pass

    return result


# ---------------------------------------------------------
# FIND LADYWOOD
# ---------------------------------------------------------

try:
    wards = get_data(WARD_URL)

    ladywood_features = []

    for feature in wards["features"]:
        attrs = feature.get("properties", {})

        name = str(attrs.get("WARDNME", "")).strip().lower()

        if name == "ladywood":
            ladywood_features.append(feature)

    if not ladywood_features:
        st.error("Ladywood ward boundary could not be found.")
        st.stop()

    ladywood_geom = unary_union(
        [shape(f["geometry"]) for f in ladywood_features]
    )

except Exception as e:
    st.error("The Ladywood boundary could not be loaded.")
    st.stop()


# ---------------------------------------------------------
# LSOAs
# ---------------------------------------------------------

try:
    lsoa_data = get_data(LSOA_URL)

    areas = []

    for feature in lsoa_data["features"]:
        if not feature.get("geometry"):
            continue

        geom = shape(feature["geometry"])

        if not geom.intersects(ladywood_geom):
            continue

        clipped = geom.intersection(ladywood_geom)

        if clipped.is_empty:
            continue

        p = feature.get("properties", {})

        areas.append({
            "Area_ID": p.get("LSOA21CD", ""),
            "Area_Name": p.get("LSOA21NM", ""),
            "CRVA_Value": p.get("MEAN"),
            "geometry": clipped
        })

    if not areas:
        st.error("No official LSOA areas were found inside Ladywood.")
        st.stop()

except Exception:
    st.error("The official LSOA data could not be loaded.")
    st.stop()


# ---------------------------------------------------------
# SPATIAL CALCULATION
# ---------------------------------------------------------

def exposure_percent(area_geom, evidence):
    total = area_geom.area

    if total <= 0:
        return 0

    exposed = 0

    for geom in evidence:
        try:
            if geom.intersects(area_geom):
                exposed += geom.intersection(area_geom).area
        except Exception:
            pass

    return min(100, (exposed / total) * 100)


# ---------------------------------------------------------
# FLOOD DATA
# ---------------------------------------------------------

try:
    flood_data = get_data(FLOOD_URL)

    flood_shapes = [
        geom for _, geom in features_to_shapes(flood_data)
        if geom.intersects(ladywood_geom)
    ]

except Exception:
    flood_shapes = []


try:
    surface_data = get_data(SURFACE_FLOOD_URL)

    surface_shapes = [
        geom for _, geom in features_to_shapes(surface_data)
        if geom.intersects(ladywood_geom)
    ]

except Exception:
    surface_shapes = []


# ---------------------------------------------------------
# BROWNFIELD DATA
# ---------------------------------------------------------

try:
    brownfield_data = get_data(BROWNFIELD_URL)

    brownfield_shapes = [
        geom for _, geom in features_to_shapes(brownfield_data)
        if geom.intersects(ladywood_geom)
    ]

except Exception:
    brownfield_shapes = []


# ---------------------------------------------------------
# CALCULATE EACH LADYWOOD AREA
# ---------------------------------------------------------

for area in areas:

    geom = area["geometry"]

    flood_zone_3 = exposure_percent(
        geom,
        flood_shapes
    )

    surface_flood = exposure_percent(
        geom,
        surface_shapes
    )

    brownfield = exposure_percent(
        geom,
        brownfield_shapes
    )

    area["Flood_Zone_3_%"] = flood_zone_3
    area["Surface_Flood_%"] = surface_flood

    # Maximum avoids counting overlapping flood evidence twice
    area["Flood_%"] = max(
        flood_zone_3,
        surface_flood
    )

    area["Brownfield_%"] = brownfield


# ---------------------------------------------------------
# NORMALISATION
# ---------------------------------------------------------

def normalise(values):

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0 for _ in values]

    return [
        (x - minimum) / (maximum - minimum)
        for x in values
    ]


crva_values = [
    float(a["CRVA_Value"])
    if a["CRVA_Value"] is not None else 0
    for a in areas
]

flood_values = [
    a["Flood_%"] for a in areas
]

brownfield_values = [
    a["Brownfield_%"] for a in areas
]

crva_norm = normalise(crva_values)
flood_norm = normalise(flood_values)
brownfield_norm = normalise(brownfield_values)


# ---------------------------------------------------------
# ENVIRONMENTAL SCREENING INDEX
# ---------------------------------------------------------

for i, area in enumerate(areas):

    area["CRVA_Index"] = crva_norm[i]
    area["Flood_Index"] = flood_norm[i]
    area["Brownfield_Index"] = brownfield_norm[i]

    area["Environmental_Index"] = (
        area["CRVA_Index"]
        + area["Flood_Index"]
        + area["Brownfield_Index"]
    ) / 3

    score = area["Environmental_Index"]

    if score >= 0.67:
        area["Risk"] = "HIGH"
    elif score >= 0.34:
        area["Risk"] = "MODERATE"
    else:
        area["Risk"] = "LOW"

    drivers = {
        "Climate vulnerability": area["CRVA_Index"],
        "Flood exposure": area["Flood_Index"],
        "Brownfield exposure": area["Brownfield_Index"]
    }

    area["Dominant_Driver"] = max(
        drivers,
        key=drivers.get
    )


# ---------------------------------------------------------
# TRAFFIC SCREENING
# ---------------------------------------------------------

try:
    road_data = get_data(ROADS_URL)

    roads = []

    for feature in road_data["features"]:

        if not feature.get("geometry"):
            continue

        geom = shape(feature["geometry"])

        if not geom.intersects(ladywood_geom):
            continue

        p = feature.get("properties", {})

        roads.append({
            "geometry": geom,
            "class": p.get("BRUM_CLASS", "Unknown"),
            "name": p.get("BRUM_ROAD_", "")
        })

except Exception:
    roads = []


for area in areas:

    nearest_class = None
    nearest_distance = float("inf")

    for road in roads:

        try:
            distance = area["geometry"].distance(
                road["geometry"]
            )

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_class = road["class"]

        except Exception:
            pass

    if nearest_distance <= 100:

        if nearest_class == "A Road":
            traffic = "HIGH"

        elif nearest_class == "B Road":
            traffic = "MODERATE"

        else:
            traffic = "LOW"

    else:
        traffic = "LOW"

    area["Traffic_Screen"] = traffic


# ---------------------------------------------------------
# TABLE
# ---------------------------------------------------------

table = pd.DataFrame([
    {
        "LSOA": a["Area_ID"],
        "Area": a["Area_Name"],
        "CRVA": round(a["CRVA_Value"], 2)
        if a["CRVA_Value"] is not None else None,
        "Flood %": round(a["Flood_%"], 2),
        "Brownfield %": round(a["Brownfield_%"], 2),
        "Environmental Index": round(
            a["Environmental_Index"], 3
        ),
        "Risk": a["Risk"],
        "Main Driver": a["Dominant_Driver"],
        "Traffic": a["Traffic_Screen"]
    }
    for a in areas
])

table = table.sort_values(
    "Environmental Index",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# MAP
# ---------------------------------------------------------

transformer = Transformer.from_crs(
    27700,
    4326,
    always_xy=True
)

lon, lat = transformer.transform(
    ladywood_geom.centroid.x,
    ladywood_geom.centroid.y
)

m = folium.Map(
    location=[lat, lon],
    zoom_start=14,
    tiles="OpenStreetMap"
)


# Ladywood boundary
folium.GeoJson(
    {
        "type": "FeatureCollection",
        "features": ladywood_features
    },
    name="Ladywood boundary",
    style_function=lambda x: {
        "color": "black",
        "weight": 3,
        "fillOpacity": 0
    }
).add_to(m)


# LSOA risk areas
lsoa_features = []

for area in areas:

    coords = []

    geom = area["geometry"]

    # Convert geometry back to GeoJSON
    from shapely.geometry import mapping

    feature = {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": {
            "LSOA": area["Area_ID"],
            "Area": area["Area_Name"],
            "Risk": area["Risk"],
            "Environmental Index": round(
                area["Environmental_Index"], 3
            ),
            "Main Driver": area["Dominant_Driver"]
        }
    }

    lsoa_features.append(feature)


def risk_style(feature):

    risk = feature["properties"]["Risk"]

    if risk == "HIGH":
        fill = "#d73027"
    elif risk == "MODERATE":
        fill = "#fc8d59"
    else:
        fill = "#91cf60"

    return {
        "color": "black",
        "weight": 1,
        "fillColor": fill,
        "fillOpacity": 0.45
    }


folium.GeoJson(
    {
        "type": "FeatureCollection",
        "features": lsoa_features
    },
    name="Environmental risk",
    style_function=risk_style,
    tooltip=folium.GeoJsonTooltip(
        fields=[
            "LSOA",
            "Area",
            "Risk",
            "Environmental Index",
            "Main Driver"
        ],
        aliases=[
            "LSOA:",
            "Area:",
            "Risk:",
            "Environmental Index:",
            "Main Driver:"
        ]
    )
).add_to(m)


# ---------------------------------------------------------
# FLOOD LAYERS
# ---------------------------------------------------------

def add_layer(data, name):

    features = []

    for feature in data["features"]:

        if not feature.get("geometry"):
            continue

        try:
            geom = shape(feature["geometry"])

            if geom.intersects(ladywood_geom):

                clipped = geom.intersection(
                    ladywood_geom
                )

                if not clipped.is_empty:

                    features.append({
                        "type": "Feature",
                        "geometry": mapping(clipped),
                        "properties": {}
                    })

        except Exception:
            pass

    if features:

        folium.GeoJson(
            {
                "type": "FeatureCollection",
                "features": features
            },
            name=name,
            style_function=lambda x: {
                "color": "blue",
                "weight": 1,
                "fillColor": "blue",
                "fillOpacity": 0.20
            }
        ).add_to(m)


try:
    add_layer(
        flood_data,
        "Flood Zone 3"
    )
except Exception:
    pass

try:
    add_layer(
        surface_data,
        "Surface flooding"
    )
except Exception:
    pass


# ---------------------------------------------------------
# BROWNFIELD
# ---------------------------------------------------------

try:

    brown_features = []

    for feature in brownfield_data["features"]:

        if not feature.get("geometry"):
            continue

        geom = shape(feature["geometry"])

        if geom.intersects(ladywood_geom):

            clipped = geom.intersection(
                ladywood_geom
            )

            if not clipped.is_empty:

                brown_features.append({
                    "type": "Feature",
                    "geometry": mapping(clipped),
                    "properties": {}
                })

    if brown_features:

        folium.GeoJson(
            {
                "type": "FeatureCollection",
                "features": brown_features
            },
            name="Brownfield",
            style_function=lambda x: {
                "color": "purple",
                "weight": 1,
                "fillColor": "purple",
                "fillOpacity": 0.30
            }
        ).add_to(m)

except Exception:
    pass


# ---------------------------------------------------------
# ROADS
# ---------------------------------------------------------

try:

    road_features = []

    for road in roads:

        try:

            clipped = road["geometry"].intersection(
                ladywood_geom
            )

            if clipped.is_empty:
                continue

            road_features.append({
                "type": "Feature",
                "geometry": mapping(clipped),
                "properties": {
                    "Class": road["class"],
                    "Road": road["name"]
                }
            })

        except Exception:
            pass

    if road_features:

        folium.GeoJson(
            {
                "type": "FeatureCollection",
                "features": road_features
            },
            name="Classified roads",
            style_function=lambda x: {
                "color": "red"
                if x["properties"]["Class"] == "A Road"
                else "blue",
                "weight": 3
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["Class", "Road"],
                aliases=["Classification:", "Road:"]
            )
        ).add_to(m)

except Exception:
    pass


# ---------------------------------------------------------
# RISK MARKERS
# ---------------------------------------------------------

for area in areas:

    point = area["geometry"].representative_point()

    x, y = transformer.transform(
        point.x,
        point.y
    )

    if area["Risk"] == "HIGH":
        colour = "red"
    elif area["Risk"] == "MODERATE":
        colour = "orange"
    else:
        colour = "green"

    folium.CircleMarker(
        location=[y, x],
        radius=6,
        color=colour,
        fill=True,
        fill_opacity=0.9,
        popup=(
            f"<b>{area['Area_Name']}</b><br>"
            f"Risk: {area['Risk']}<br>"
            f"Index: {area['Environmental_Index']:.3f}<br>"
            f"Driver: {area['Dominant_Driver']}"
        )
    ).add_to(m)


# ---------------------------------------------------------
# MAP LEGEND
# ---------------------------------------------------------

legend = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background: white;
padding: 10px;
border: 2px solid grey;
font-size: 13px;
">
<b>Risk</b><br>
<span style="color:red;">●</span> High<br>
<span style="color:orange;">●</span> Moderate<br>
<span style="color:green;">●</span> Low
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend)
)

folium.LayerControl().add_to(m)

components.html(
    m.get_root().render(),
    height=650,
    scrolling=False
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

st.subheader("Environmental Screening Classification")

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# CHARTS
# ---------------------------------------------------------

st.subheader("Environmental Risk by Ladywood Area")

chart = table.set_index("Area")[
    ["Environmental Index"]
]

st.bar_chart(chart)


st.subheader("Flood Exposure by Ladywood Area")

flood_chart = table.set_index("Area")[
    ["Flood %"]
]

st.bar_chart(flood_chart)


st.subheader("Brownfield Exposure by Ladywood Area")

brown_chart = table.set_index("Area")[
    ["Brownfield %"]
]

st.bar_chart(brown_chart)


# ---------------------------------------------------------
# PRIORITY AREAS
# ---------------------------------------------------------

st.subheader("Priority Ladywood Areas for Further Investigation")

priority = table[
    table["Risk"].isin(["HIGH", "MODERATE"])
]

st.dataframe(
    priority,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------

st.download_button(
    "Download Ladywood screening results",
    table.to_csv(index=False),
    "Ladywood_environmental_screening.csv",
    "text/csv"
)
