import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt

from shapely.geometry import shape
from streamlit_folium import st_folium


# --------------------------------------------------
# 1. PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Dashboard")
st.write(
    "Environmental factors for Ladywood, Birmingham, "
    "using Birmingham City Council's CRVA data."
)

st.info(
    "The council's CRVA uses a 100 m grid and factor-specific "
    "standardisation. Individual source layers below are displayed "
    "separately; they are not presented as recalculated official "
    "CRVA scores."
)


# --------------------------------------------------
# 2. OFFICIAL DATA SOURCES
# --------------------------------------------------

CRVA_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

BROWNFIELD_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_LandUse/MapServer"
)


# Council CRVA source layers
LAYERS = {
    "Official CRVA score by LSOA": (CRVA_URL, 15),
    "Fluvial flooding": (CRVA_URL, 109),
    "Surface flooding": (CRVA_URL, 1546),
    "Surface temperature (1981–2000)": (CRVA_URL, 18),
    "Local climate zones": (CRVA_URL, 4),
    "Greenspace": (CRVA_URL, 1549),
    "Woodland / tree cover context": (CRVA_URL, 1552),
    "Tree canopy layer": (CRVA_URL, 3),
}


# --------------------------------------------------
# 3. DOWNLOAD FEATURES FROM ARCGIS
# --------------------------------------------------

@st.cache_data(ttl=3600)
def download_layer(service_url, layer_id):
    """
    Download a feature layer from the council's ArcGIS service.
    Returns a GeoDataFrame or an empty GeoDataFrame if unavailable.
    """

    query_url = f"{service_url}/{layer_id}/query"

    parameters = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 2000
    }

    try:
        response = requests.get(
            query_url,
            params=parameters,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        if "features" not in data or not data["features"]:
            return gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326"
            )

        records = []

        for feature in data["features"]:
            properties = feature.get("properties", {})
            geometry_data = feature.get("geometry")

            if geometry_data:
                properties["geometry"] = shape(geometry_data)
                records.append(properties)

        if not records:
            return gpd.GeoDataFrame(
                geometry=[],
                crs="EPSG:4326"
            )

        return gpd.GeoDataFrame(
            records,
            geometry="geometry",
            crs="EPSG:4326"
        )

    except Exception as error:
        st.warning(
            f"Could not download layer {layer_id}: {error}"
        )
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )


# --------------------------------------------------
# 4. GET LADYWOOD BOUNDARY
# --------------------------------------------------

@st.cache_data(ttl=3600)
def get_ladywood_boundary():
    """
    Find the official Ladywood ward polygon.
    """

    wards = download_layer(CRVA_URL, 14)

    if wards.empty:
        return None

    # Find the ward-name field without assuming its capitalisation.
    name_fields = [
        column for column in wards.columns
        if column.upper() in ["WARDNME", "WARD_NAME", "NAME"]
    ]

    if not name_fields:
        return None

    name_field = name_fields[0]

    ladywood = wards[
        wards[name_field].astype(str).str.strip().str.lower()
        == "ladywood"
    ]

    if ladywood.empty:
        return None

    return ladywood[["geometry"]].dissolve()


boundary = get_ladywood_boundary()

if boundary is None:
    st.error(
        "The Ladywood ward boundary could not be loaded. "
        "Check the council service connection and ward-name field."
    )
    st.stop()


# --------------------------------------------------
# 5. CLIP DATA TO LADYWOOD
# --------------------------------------------------

def clip_to_ladywood(data):
    """
    Keep features that intersect Ladywood.
    Uses the official ward polygon as the study boundary.
    """

    if data.empty:
        return data

    try:
        boundary_same_crs = boundary.to_crs(data.crs)

        clipped = gpd.clip(data, boundary_same_crs)

        return clipped[~clipped.geometry.is_empty]

    except Exception as error:
        st.warning(f"Could not clip layer to Ladywood: {error}")
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )


# --------------------------------------------------
# 6. SELECT A FACTOR
# --------------------------------------------------

factor_name = st.selectbox(
    "Choose an environmental factor",
    list(LAYERS.keys())
)

service_url, layer_id = LAYERS[factor_name]

with st.spinner(f"Loading {factor_name}..."):
    factor_data = download_layer(service_url, layer_id)
    factor_data = clip_to_ladywood(factor_data)


# --------------------------------------------------
# 7. DRAW THE MAP
# --------------------------------------------------

map_centre = [52.48, -1.92]

environment_map = folium.Map(
    location=map_centre,
    zoom_start=13,
    tiles="OpenStreetMap"
)

# Draw the Ladywood boundary.
folium.GeoJson(
    boundary.to_crs("EPSG:4326").to_json(),
    name="Ladywood boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 2,
        "fillOpacity": 0
    }
).add_to(environment_map)


if not factor_data.empty:

    # Create a readable popup from available attribute fields.
    possible_fields = [
        column for column in factor_data.columns
        if column != "geometry"
    ]

    def make_popup(properties):
        lines = []

        for field in possible_fields[:8]:
            value = properties.get(field)

            if value is not None:
                lines.append(f"{field}: {value}")

        return "<br>".join(lines) or factor_name

    folium.GeoJson(
        factor_data.to_crs("EPSG:4326").to_json(),
        name=factor_name,
        style_function=lambda feature: {
            "color": "#6b3f1d",
            "weight": 1,
            "fillColor": "#d98c42",
            "fillOpacity": 0.45
        },
        tooltip=folium.GeoJsonTooltip(
            fields=possible_fields[:1],
            aliases=possible_fields[:1],
            sticky=True
        ) if possible_fields else None,
        popup=folium.GeoJsonPopup(
            fields=possible_fields[:min(8, len(possible_fields))]
        ) if possible_fields else None
    ).add_to(environment_map)

    st.success(
        f"Loaded {len(factor_data)} feature(s) for this layer."
    )

else:
    st.warning(
        "No features were returned for this layer. "
        "This does not mean the environmental risk is zero."
    )

folium.LayerControl().add_to(environment_map)

st.subheader(f"Map: {factor_name}")

st_folium(
    environment_map,
    use_container_width=True,
    height=600
)


# --------------------------------------------------
# 8. SHOW AVAILABLE DATA
# --------------------------------------------------

st.subheader("Data table")

if not factor_data.empty:
    table = factor_data.drop(
        columns="geometry",
        errors="ignore"
    )

    st.dataframe(table, use_container_width=True)

    st.download_button(
        "Download this factor as CSV",
        data=table.to_csv(index=False),
        file_name="ladywood_factor_data.csv",
        mime="text/csv"
    )

else:
    st.write("No data table is available for this selection.")


# --------------------------------------------------
# 9. SIMPLE CHART OF NUMERIC ATTRIBUTES
# --------------------------------------------------

st.subheader("Explore numeric attributes")

if not factor_data.empty:
    table = factor_data.drop(
        columns="geometry",
        errors="ignore"
    )

    numeric_columns = table.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        selected_column = st.selectbox(
            "Choose a numeric field to chart",
            numeric_columns
        )

        chart_data = pd.to_numeric(
            table[selected_column],
            errors="coerce"
        ).dropna()

        if not chart_data.empty:
            fig, ax = plt.subplots()

            ax.hist(chart_data, bins=15)
            ax.set_title(
                f"{selected_column} — {factor_name}"
            )
            ax.set_xlabel(selected_column)
            ax.set_ylabel("Number of features")

            st.pyplot(fig)

            st.write(
                "This chart shows the distribution of values "
                "returned by the source layer. It is not, by itself, "
                "a standardised CRVA risk score."
            )
        else:
            st.write("No numeric values are available in this field.")
    else:
        st.write(
            "This layer has no numeric attributes suitable "
            "for this simple chart."
        )


# --------------------------------------------------
# 10. METHODOLOGY NOTES
# --------------------------------------------------

with st.expander("Methodology and limitations"):
    st.markdown("""
    **Official reference:** Birmingham City Council's Climate Risk
    and Vulnerability Assessment (CRVA).

    - The council's method uses a 100 m grid.
    - Each factor is standardised to a score between 0 and 1.
    - Factor-specific transformations are used; raw values should not
      be treated as standardised scores.
    - The full official CRVA combines 11 factors with equal weighting.
    - This dashboard focuses on factors 3–9 and shows the source layers
      separately for investigation.
    - The published temperature layer represents 1981–2000 and is not
      current temperature data.
    - The greenspace, woodland and tree-canopy layers are source layers;
      they do not necessarily represent the council's calculated deficit
      scores directly.
    - Missing data are not treated as zero.
    - Brownfield land is an additional project indicator, not one of the
      11 official CRVA factors.
    """)
    
    st.markdown(
        "[Open Birmingham City Council's CRVA methodology]"
        "(https://maps.birmingham.gov.uk/webapps/crva/crva.html)"
    )
