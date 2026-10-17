import streamlit as st
import requests
import pandas as pd
import geopandas as gpd
import folium
import matplotlib.pyplot as plt

from shapely.geometry import shape
from streamlit_folium import st_folium


# =========================================================
# 1. PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Dashboard")
st.caption("Ladywood, Birmingham, United Kingdom")

st.info(
    "This dashboard uses Birmingham City Council's CRVA service. "
    "Source layers are shown separately. Only the council's published "
    "CRVA layer is labelled as the official combined CRVA score."
)


# =========================================================
# 2. OFFICIAL COUNCIL SERVICES
# =========================================================

CRVA_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

LANDUSE_URL = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_LandUse/MapServer"
)

# The official CRVA layer IDs.
# Some layers are feature layers; others may be image/raster layers.

FACTORS = {
    "Official CRVA score by LSOA": {
        "url": CRVA_URL,
        "id": 15,
        "type": "features"
    },
    "3. Fluvial flooding": {
        "url": CRVA_URL,
        "id": 109,
        "type": "features"
    },
    "4. Surface flooding": {
        "url": CRVA_URL,
        "id": 1546,
        "type": "features"
    },
    "5. Surface temperature (1981–2000)": {
        "url": CRVA_URL,
        "id": 18,
        "type": "image"
    },
    "6. Local climate zones": {
        "url": CRVA_URL,
        "id": 4,
        "type": "features"
    },
    "7–8. Greenspace source layer": {
        "url": CRVA_URL,
        "id": 1549,
        "type": "features"
    },
    "9. Tree canopy source layer": {
        "url": CRVA_URL,
        "id": 3,
        "type": "image"
    },
    "Woodland source layer": {
        "url": CRVA_URL,
        "id": 1552,
        "type": "features"
    },
    "Brownfield register (additional indicator)": {
        "url": LANDUSE_URL,
        "id": 38,
        "type": "features"
    }
}


# =========================================================
# 3. DOWNLOAD A FEATURE LAYER
# =========================================================

@st.cache_data(ttl=3600)
def download_features(service_url, layer_id, where="1=1"):
    """Download features from an ArcGIS layer."""

    query_url = f"{service_url}/{layer_id}/query"

    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultRecordCount": 2000
    }

    response = requests.get(
        query_url,
        params=params,
        timeout=60
    )
    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise ValueError(data["error"].get("message", "ArcGIS query failed"))

    features = data.get("features", [])

    records = []

    for feature in features:
        geometry_data = feature.get("geometry")

        if geometry_data:
            record = feature.get("properties", {}).copy()
            record["geometry"] = shape(geometry_data)
            records.append(record)

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


# =========================================================
# 4. LOAD THE LADYWOOD WARD BOUNDARY
# =========================================================

@st.cache_data(ttl=3600)
def get_ladywood_boundary():
    """
    The CRVA By Ward layer contains the ward name and geometry.
    Query Ladywood directly instead of guessing a boundary.
    """

    ward_data = download_features(
        CRVA_URL,
        12,
        "WARDNME = 'Ladywood'"
    )

    if ward_data.empty:
        return None

    return ward_data[["geometry"]].dissolve()


try:
    boundary = get_ladywood_boundary()

except Exception as error:
    st.error(
        "The official Ladywood boundary could not be downloaded. "
        "Check the app logs and internet connection."
    )
    st.code(str(error))
    st.stop()

if boundary is None or boundary.empty:
    st.error(
        "The council service returned no Ladywood boundary. "
        "The dashboard cannot safely filter data to Ladywood."
    )
    st.stop()


# =========================================================
# 5. CLIP FEATURE DATA TO LADYWOOD
# =========================================================

def clip_to_ladywood(data):
    """Keep features intersecting the Ladywood ward boundary."""

    if data is None or data.empty:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    try:
        ward = boundary.to_crs(data.crs)
        return gpd.clip(data, ward)

    except Exception as error:
        st.warning(f"Could not clip this layer: {error}")

        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )


# =========================================================
# 6. FACTOR SELECTION
# =========================================================

factor_name = st.selectbox(
    "Choose an environmental factor",
    list(FACTORS.keys())
)

factor = FACTORS[factor_name]

st.subheader(factor_name)

st.write(
    f"Council service layer ID: {factor['id']}"
)

if factor_name == "5. Surface temperature (1981–2000)":
    st.warning(
        "This is historical temperature data from 1981–2000, "
        "not a current temperature measurement."
    )

if factor_name == "9. Tree canopy source layer":
    st.info(
        "The council describes this as a vegetation-model hillshade "
        "showing tree canopy cover. It may need to be displayed as a "
        "map image rather than downloaded as individual features."
    )

if factor_name == "7–8. Greenspace source layer":
    st.warning(
        "This is the source greenspace layer. It is not itself the "
        "official open-greenspace deficit score or other-greenspace "
        "deficit score. Those require the council's calculation method."
    )


# =========================================================
# 7. BUILD THE MAP
# =========================================================

map_centre = [52.48, -1.92]

environment_map = folium.Map(
    location=map_centre,
    zoom_start=13,
    tiles="OpenStreetMap"
)

folium.GeoJson(
    boundary.to_crs("EPSG:4326").to_json(),
    name="Ladywood ward boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 3,
        "fillOpacity": 0
    },
    tooltip="Ladywood ward"
).add_to(environment_map)


# =========================================================
# 8. ADD THE SELECTED FEATURE LAYER
# =========================================================

factor_data = gpd.GeoDataFrame(
    geometry=[],
    crs="EPSG:4326"
)

if factor["type"] == "features":

    try:
        with st.spinner(f"Loading {factor_name}..."):
            all_features = download_features(
                factor["url"],
                factor["id"]
            )

            factor_data = clip_to_ladywood(all_features)

        if not factor_data.empty:

            fields = [
                column for column in factor_data.columns
                if column != "geometry"
            ]

            # Use available fields for map popups.
            popup_fields = fields[:8]

            folium.GeoJson(
                factor_data.to_crs("EPSG:4326").to_json(),
                name=factor_name,
                style_function=lambda feature: {
                    "color": "#754c24",
                    "weight": 1,
                    "fillColor": "#e6a04b",
                    "fillOpacity": 0.45
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=fields[:1],
                    aliases=fields[:1]
                ) if fields else None,
                popup=folium.GeoJsonPopup(
                    fields=popup_fields
                ) if popup_fields else None
            ).add_to(environment_map)

            st.success(
                f"Retrieved {len(factor_data)} feature(s) "
                "intersecting Ladywood."
            )

        else:
            st.warning(
                "The query returned no features in Ladywood. "
                "This does not mean the environmental risk is zero."
            )

    except Exception as error:
        st.error(
            "This feature layer could not be retrieved. "
            "The rest of the dashboard remains available."
        )
        st.code(str(error))

else:
    # Raster/image layers are not assumed to be downloadable as polygons.
    # Add a direct link to the exact council layer.
    layer_page = f"{factor['url']}/{factor['id']}"

    st.warning(
        "This is treated as an image layer. The script will not invent "
        "polygon data or numeric values for it."
    )

    st.link_button(
        "Open this official council layer",
        layer_page
    )


folium.LayerControl(collapsed=False).add_to(environment_map)

st_folium(
    environment_map,
    use_container_width=True,
    height=600
)


# =========================================================
# 9. SHOW DATA TABLE
# =========================================================

st.subheader("Retrieved source data")

if not factor_data.empty:

    table = factor_data.drop(
        columns="geometry",
        errors="ignore"
    )

    st.dataframe(
        table,
        use_container_width=True
    )

    st.download_button(
        label="Download source data as CSV",
        data=table.to_csv(index=False),
        file_name="ladywood_environmental_factor.csv",
        mime="text/csv"
    )

else:
    st.write(
        "No feature table is available for this selection. "
        "Image layers are linked to the official council service."
    )


# =========================================================
# 10. SIMPLE CHART
# =========================================================

st.subheader("Explore available numeric data")

if not factor_data.empty:

    table = factor_data.drop(
        columns="geometry",
        errors="ignore"
    )

    numeric_fields = table.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_fields:

        chosen_field = st.selectbox(
            "Choose a numeric field",
            numeric_fields
        )

        values = pd.to_numeric(
            table[chosen_field],
            errors="coerce"
        ).dropna()

        if not values.empty:

            fig, ax = plt.subplots()

            ax.hist(values, bins=15)

            ax.set_title(
                f"{chosen_field}: {factor_name}"
            )
            ax.set_xlabel(chosen_field)
            ax.set_ylabel("Number of features")

            st.pyplot(fig)
            plt.close(fig)

            st.write(
                "This histogram shows the distribution of the selected "
                "source field. It is not automatically a CRVA risk score."
            )

        else:
            st.info("This field has no usable numeric values.")

    else:
        st.info(
            "This layer has no numeric fields suitable for a histogram."
        )


# =========================================================
# 11. OFFICIAL CRVA REFERENCE
# =========================================================

with st.expander("CRVA methodology and data limitations"):

    st.markdown("""
    **Official method**

    Birmingham's CRVA standardises environmental and social factors
    onto a common 100 m grid. Each factor receives a score between
    0 and 1, where higher scores indicate greater risk or vulnerability.
    The official combined assessment sums the factor scores using
    equal weighting.

    **What this dashboard currently does**

    - Uses the council's official ward boundary to define Ladywood.
    - Displays available feature layers separately.
    - Provides the published CRVA-by-LSOA layer as a reference.
    - Does not turn missing data into zero.
    - Does not label raw source values as official standardised scores.
    - Keeps brownfield land as an additional project indicator.

    **Important limitations**

    - The full official CRVA uses 11 factors. This dashboard focuses on
      factors 3–9 and does not recalculate the complete official score.
    - The greenspace source layer alone does not separate both deficit
      factors.
    - Temperature data are from 1981–2000.
    - Tree-canopy and temperature layers may require image-based GIS
      methods rather than ordinary feature queries.
    """)

    st.markdown(
        "[Open the council's CRVA map]"
        "(https://maps.birmingham.gov.uk/webapps/crva/crva.html)"
    )

    st.markdown(
        "[Open the official CRVA data catalogue]"
        "(https://maps.birmingham.gov.uk/server/rest/services/"
        "CRVA/CRVA_2025/MapServer)"
    )

    st.markdown(
        "[Open the CRVA methodology PDF]"
        "(https://research.birmingham.ac.uk/files/253369474/"
        "CRVA_methodology_guidancedoc_v1.1.pdf)"
    )
