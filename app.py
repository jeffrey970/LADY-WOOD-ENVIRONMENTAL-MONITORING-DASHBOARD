import streamlit as st
import pandas as pd
import requests
import folium
import streamlit.components.v1 as components
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import Transformer

# =========================================================
# LADYWOOD ENVIRONMENTAL DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Dashboard")
st.caption("Environmental screening of Ladywood using official Birmingham spatial data.")

# =========================================================
# OFFICIAL BIRMINGHAM DATA
# =========================================================

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

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

# =========================================================
# LOAD ARCGIS DATA
# =========================================================

def load_layer(url):
    response = requests.get(
        url,
        params={
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson"
        },
        timeout=90
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:
        raise ValueError("No features returned.")

    return data


def get_geometries(data):
    geometries = []

    for feature in data.get("features", []):
        try:
            if feature.get("geometry"):
                geometries.append(
                    shape(feature["geometry"])
                )
        except Exception:
            pass

    return geometries


# =========================================================
# LADYWOOD BOUNDARY
# =========================================================

try:

    ward_data = load_layer(WARD_URL)

    ladywood = []

    for feature in ward_data["features"]:

        properties = feature.get("properties", {})

        match = False

        for value in properties.values():

            if value is not None:

                if str(value).strip().lower() == "ladywood":
                    match = True
                    break

        if match and feature.get("geometry"):
            ladywood.append(feature)

    if not ladywood:

        st.error(
            "The official Birmingham ward layer did not return Ladywood."
        )
        st.stop()

    ladywood_geometry = unary_union(
        [
            shape(feature["geometry"])
            for feature in ladywood
        ]
    )

except Exception:

    st.error(
        "The official Ladywood boundary could not be loaded."
    )
    st.stop()


# =========================================================
# OFFICIAL LSOAs
# =========================================================

try:

    lsoa_data = load_layer(LSOA_URL)

    areas = []

    for feature in lsoa_data["features"]:

        if not feature.get("geometry"):
            continue

        try:
            geometry = shape(feature["geometry"])

            if not geometry.intersects(ladywood_geometry):
                continue

            clipped = geometry.intersection(
                ladywood_geometry
            )

            if clipped.is_empty:
                continue

            properties = feature.get("properties", {})

            areas.append({
                "LSOA": properties.get("LSOA21CD", ""),
                "Area": properties.get("LSOA21NM", ""),
                "CRVA": properties.get("MEAN"),
                "geometry": clipped
            })

        except Exception:
            continue

    if not areas:
        st.error(
            "No official LSOA areas were found within Ladywood."
        )
        st.stop()

except Exception:

    st.error(
        "The official LSOA data could not be loaded."
    )
    st.stop()


# =========================================================
# EXPOSURE CALCULATION
# =========================================================

def exposure(area_geometry, evidence):

    total_area = area_geometry.area

    if total_area == 0:
        return 0

    exposed_area = 0

    for evidence_geometry in evidence:

        try:

            if evidence_geometry.intersects(area_geometry):

                exposed_area += (
                    evidence_geometry
                    .intersection(area_geometry)
                    .area
                )

        except Exception:
            pass

    percentage = (
        exposed_area / total_area
    ) * 100

    return min(percentage, 100)


# =========================================================
# FLOOD DATA
# =========================================================

try:

    flood_data = load_layer(FLOOD_URL)

    flood_geometry = [
        geometry
        for geometry in get_geometries(flood_data)
        if geometry.intersects(ladywood_geometry)
    ]

except Exception:

    flood_geometry = []


try:

    surface_data = load_layer(SURFACE_FLOOD_URL)

    surface_geometry = [
        geometry
        for geometry in get_geometries(surface_data)
        if geometry.intersects(ladywood_geometry)
    ]

except Exception:

    surface_geometry = []


# =========================================================
# BROWNFIELD DATA
# =========================================================

try:

    brownfield_data = load_layer(BROWNFIELD_URL)

    brownfield_geometry = [
        geometry
        for geometry in get_geometries(brownfield_data)
        if geometry.intersects(ladywood_geometry)
    ]

except Exception:

    brownfield_geometry = []


# =========================================================
# CALCULATE ENVIRONMENTAL EXPOSURE
# =========================================================

for area in areas:

    geometry = area["geometry"]

    flood_zone_3 = exposure(
        geometry,
        flood_geometry
    )

    surface_flood = exposure(
        geometry,
        surface_geometry
    )

    brownfield = exposure(
        geometry,
        brownfield_geometry
    )

    area["Flood Zone 3 %"] = flood_zone_3
    area["Surface Flood %"] = surface_flood

    # Avoid double-counting overlapping flood information
    area["Flood %"] = max(
        flood_zone_3,
        surface_flood
    )

    area["Brownfield %"] = brownfield


# =========================================================
# NORMALISATION
# =========================================================

def normalise(values):

    values = [float(v) for v in values]

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [0] * len(values)

    return [
        (value - minimum) /
        (maximum - minimum)
        for value in values
    ]


crva_values = [
    float(area["CRVA"])
    if area["CRVA"] is not None
    else 0
    for area in areas
]

flood_values = [
    area["Flood %"]
    for area in areas
]

brownfield_values = [
    area["Brownfield %"]
    for area in areas
]

crva_index = normalise(crva_values)
flood_index = normalise(flood_values)
brownfield_index = normalise(
    brownfield_values
)


# =========================================================
# ENVIRONMENTAL INDEX
# =========================================================

for i, area in enumerate(areas):

    area["CRVA Index"] = crva_index[i]
    area["Flood Index"] = flood_index[i]
    area["Brownfield Index"] = brownfield_index[i]

    area["Environmental Index"] = (
        area["CRVA Index"]
        + area["Flood Index"]
        + area["Brownfield Index"]
    ) / 3

    score = area["Environmental Index"]

    if score >= 0.67:
        area["Risk"] = "HIGH"

    elif score >= 0.34:
        area["Risk"] = "MODERATE"

    else:
        area["Risk"] = "LOW"

    drivers = {
        "Climate vulnerability": area["CRVA Index"],
        "Flood exposure": area["Flood Index"],
        "Brownfield exposure": area["Brownfield Index"]
    }

    area["Main Driver"] = max(
        drivers,
        key=drivers.get
    )


# =========================================================
# ROAD DATA
# =========================================================

try:

    road_data = load_layer(ROADS_URL)

    roads = []

    for feature in road_data["features"]:

        if not feature.get("geometry"):
            continue

        try:

            geometry = shape(
                feature["geometry"]
            )

            if not geometry.intersects(
                ladywood_geometry
            ):
                continue

            properties = feature.get(
                "properties",
                {}
            )

            roads.append({
                "geometry": geometry,
                "class": properties.get(
                    "BRUM_CLASS",
                    "Unknown"
                ),
                "name": properties.get(
                    "BRUM_ROAD_",
                    ""
                )
            })

        except Exception:
            continue

except Exception:

    roads = []


# =========================================================
# TRAFFIC SCREENING
# =========================================================

for area in areas:

    nearest_distance = float("inf")
    nearest_class = ""

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
            area["Traffic"] = "HIGH"

        elif nearest_class == "B Road":
            area["Traffic"] = "MODERATE"

        else:
            area["Traffic"] = "LOW"

    else:

        area["Traffic"] = "LOW"


# =========================================================
# RESULTS TABLE
# =========================================================

results = pd.DataFrame([

    {
        "LSOA": area["LSOA"],
        "Area": area["Area"],
        "CRVA": round(
            area["CRVA"], 2
        ) if area["CRVA"] is not None else None,
        "Flood %": round(
            area["Flood %"], 2
        ),
        "Brownfield %": round(
            area["Brownfield %"], 2
        ),
        "Environmental Index": round(
            area["Environmental Index"],
            3
        ),
        "Risk": area["Risk"],
        "Main Driver": area["Main Driver"],
        "Traffic": area["Traffic"]
    }

    for area in areas

])

results = results.sort_values(
    "Environmental Index",
    ascending=False
).reset_index(drop=True)


# =========================================================
# MAP
# =========================================================

transformer = Transformer.from_crs(
    27700,
    4326,
    always_xy=True
)

centre = ladywood_geometry.centroid

longitude, latitude = transformer.transform(
    centre.x,
    centre.y
)

m = folium.Map(
    location=[
        latitude,
        longitude
    ],
    zoom_start=14,
    tiles="OpenStreetMap"
)


# =========================================================
# LADYWOOD BOUNDARY
# =========================================================

folium.GeoJson(
    {
        "type": "FeatureCollection",
        "features": ladywood
    },
    name="Ladywood boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 3,
        "fillOpacity": 0
    }
).add_to(m)


# =========================================================
# LSOA RISK
# =========================================================

risk_features = []

for area in areas:

    risk_features.append({

        "type": "Feature",

        "geometry": mapping(
            area["geometry"]
        ),

        "properties": {

            "LSOA": area["LSOA"],

            "Area": area["Area"],

            "Risk": area["Risk"],

            "Index": round(
                area["Environmental Index"],
                3
            ),

            "Driver": area["Main Driver"]
        }
    })


def risk_style(feature):

    risk = feature[
        "properties"
    ]["Risk"]

    if risk == "HIGH":
        colour = "#d73027"

    elif risk == "MODERATE":
        colour = "#fc8d59"

    else:
        colour = "#91cf60"

    return {
        "color": "black",
        "weight": 1,
        "fillColor": colour,
        "fillOpacity": 0.45
    }


folium.GeoJson(

    {
        "type": "FeatureCollection",
        "features": risk_features
    },

    name="Environmental risk",

    style_function=risk_style,

    tooltip=folium.GeoJsonTooltip(

        fields=[
            "LSOA",
            "Area",
            "Risk",
            "Index",
            "Driver"
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


# =========================================================
# ADD FLOOD / BROWNFIELD MAP LAYERS
# =========================================================

def add_evidence_layer(
    geometries,
    name,
    colour
):

    features = []

    for geometry in geometries:

        try:

            clipped = geometry.intersection(
                ladywood_geometry
            )

            if not clipped.is_empty:

                features.append({

                    "type": "Feature",

                    "geometry": mapping(
                        clipped
                    ),

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

            style_function=lambda feature,
            colour=colour: {

                "color": colour,
                "weight": 1,
                "fillColor": colour,
                "fillOpacity": 0.20
            }

        ).add_to(m)


add_evidence_layer(
    flood_geometry,
    "Flood Zone 3",
    "blue"
)

add_evidence_layer(
    surface_geometry,
    "Surface flooding",
    "cyan"
)

add_evidence_layer(
    brownfield_geometry,
    "Brownfield",
    "purple"
)


# =========================================================
# CLASSIFIED ROADS
# =========================================================

road_features = []

for road in roads:

    try:

        clipped = road["geometry"].intersection(
            ladywood_geometry
        )

        if clipped.is_empty:
            continue

        road_features.append({

            "type": "Feature",

            "geometry": mapping(
                clipped
            ),

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

        style_function=lambda feature: {

            "color":
                "red"
                if feature["properties"]["Class"]
                == "A Road"
                else "blue",

            "weight": 3
        },

        tooltip=folium.GeoJsonTooltip(

            fields=[
                "Class",
                "Road"
            ],

            aliases=[
                "Classification:",
                "Road:"
            ]
        )

    ).add_to(m)


# =========================================================
# RISK MARKERS
# =========================================================

for area in areas:

    point = area[
        "geometry"
    ].representative_point()

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

        location=[
            y,
            x
        ],

        radius=6,

        color=colour,

        fill=True,

        fill_opacity=0.9,

        popup=(
            f"<b>{area['Area']}</b><br>"
            f"Risk: {area['Risk']}<br>"
            f"Index: "
            f"{area['Environmental Index']:.3f}<br>"
            f"Driver: {area['Main Driver']}"
        )

    ).add_to(m)


# =========================================================
# MAP LEGEND
# =========================================================

legend = """

<div style="
position: fixed;
bottom: 25px;
left: 25px;
z-index: 9999;
background-color: white;
padding: 10px;
border: 2px solid grey;
font-size: 13px;
">

<b>Environmental Risk</b><br>

<span style="color:red;">●</span>
High<br>

<span style="color:orange;">●</span>
Moderate<br>

<span style="color:green;">●</span>
Low

</div>

"""

m.get_root().html.add_child(
    folium.Element(legend)
)

folium.LayerControl().add_to(m)


# =========================================================
# DISPLAY MAP
# =========================================================

components.html(
    m.get_root().render(),
    height=650,
    scrolling=False
)


# =========================================================
# DASHBOARD RESULTS
# =========================================================

st.subheader(
    "Environmental Screening Classification"
)

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)


st.subheader(
    "Environmental Risk by Ladywood Area"
)

st.bar_chart(
    results.set_index("Area")[
        ["Environmental Index"]
    ]
)


st.subheader(
    "Flood Exposure by Ladywood Area"
)

st.bar_chart(
    results.set_index("Area")[
        ["Flood %"]
    ]
)


st.subheader(
    "Brownfield Exposure by Ladywood Area"
)

st.bar_chart(
    results.set_index("Area")[
        ["Brownfield %"]
    ]
)


# =========================================================
# PRIORITY AREAS
# =========================================================

st.subheader(
    "Priority Ladywood Areas for Further Investigation"
)

priority = results[
    results["Risk"].isin(
        ["HIGH", "MODERATE"]
    )
]

st.dataframe(
    priority,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# DOWNLOAD
# =========================================================

st.download_button(
    "Download screening results",
    results.to_csv(index=False),
    "Ladywood_environmental_screening.csv",
    "text/csv"
)
