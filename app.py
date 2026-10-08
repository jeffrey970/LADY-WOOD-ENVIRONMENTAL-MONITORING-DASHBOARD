import io
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
import plotly.express as px

from shapely.geometry import shape
from streamlit_folium import st_folium


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Monitoring Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("Ladywood Environmental Monitoring Risk Dashboard")

st.caption(
    "Ladywood-only environmental screening using official "
    "Birmingham City Council spatial data and DEFRA air-quality data."
)


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

CRVA_SERVICE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = f"{CRVA_SERVICE}/14"
LSOA_CRVA_LAYER = f"{CRVA_SERVICE}/15"

FLOOD_ZONE_3_LAYER = f"{CRVA_SERVICE}/109"
SURFACE_FLOOD_LAYER = f"{CRVA_SERVICE}/1546"

BROWNFIELD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

ROAD_CLASS_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Highways_open/MapServer/26"
)

DEFRA_BASE = (
    "https://uk-air.defra.gov.uk/datastore/data_files/site_data/"
)

DEFRA_SITE = "BMLD"

AIR_YEARS = list(range(2019, 2027))


# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def repair_geometry(gdf):
    if gdf.empty:
        return gdf

    gdf = gdf.copy()

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)

    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    return gdf


def arcgis_query(
    layer_url,
    where="1=1",
    fields="*",
    bbox=None
):
    params = {
        "where": where,
        "outFields": fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox

        params["geometry"] = (
            f"{xmin},{ymin},{xmax},{ymax}"
        )
        params["geometryType"] = "esriGeometryEnvelope"
        params["inSR"] = "27700"
        params["spatialRel"] = "esriSpatialRelIntersects"

    response = requests.get(
        f"{layer_url}/query",
        params=params,
        timeout=90
    )

    response.raise_for_status()

    result = response.json()

    if "features" not in result:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    rows = []

    for feature in result["features"]:
        geometry_data = feature.get("geometry")

        if geometry_data is None:
            continue

        properties = feature.get(
            "properties",
            {}
        )

        properties["geometry"] = shape(
            geometry_data
        )

        rows.append(properties)

    if not rows:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:4326"
        )

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return repair_geometry(gdf)


def get_bbox(gdf):
    bounds = gdf.total_bounds

    return (
        float(bounds[0]),
        float(bounds[1]),
        float(bounds[2]),
        float(bounds[3])
    )


# ============================================================
# LADYWOOD BOUNDARY
# ============================================================

@st.cache_data(ttl=3600)
def load_wards():

    return arcgis_query(
        WARD_LAYER,
        fields="*"
    )


def get_ladywood():

    wards = load_wards()

    if wards.empty:
        raise RuntimeError(
            "The official Birmingham ward layer returned no data."
        )

    wards = wards.to_crs("EPSG:27700")

    possible_columns = [
        column
        for column in wards.columns
        if "WARDNME" in str(column).upper()
    ]

    if not possible_columns:
        raise RuntimeError(
            "The official ward name field could not be found."
        )

    ward_name_column = possible_columns[0]

    ladywood = wards[
        wards[ward_name_column]
        .astype(str)
        .str.contains(
            "Ladywood",
            case=False,
            na=False
        )
    ].copy()

    if ladywood.empty:
        raise RuntimeError(
            "Ladywood could not be found in the official "
            "Birmingham ward boundary data."
        )

    return repair_geometry(ladywood)


# ============================================================
# LSOA + CRVA
# ============================================================

def get_lsoa_crva(ladywood):

    bbox = get_bbox(ladywood)

    lsoa = arcgis_query(
        LSOA_CRVA_LAYER,
        fields="*",
        bbox=bbox
    )

    if lsoa.empty:
        raise RuntimeError(
            "The official CRVA LSOA layer returned no data."
        )

    lsoa = lsoa.to_crs("EPSG:27700")

    ladywood_union = ladywood.geometry.unary_union

    lsoa = lsoa[
        lsoa.geometry.intersects(
            ladywood_union
        )
    ].copy()

    if lsoa.empty:
        raise RuntimeError(
            "No LSOAs intersect the Ladywood boundary."
        )

    lsoa["geometry"] = (
        lsoa.geometry
        .intersection(ladywood_union)
    )

    return repair_geometry(lsoa)


# ============================================================
# FLOOD LAYERS
# ============================================================

def get_spatial_layer(
    layer_url,
    ladywood
):

    bbox = get_bbox(ladywood)

    layer = arcgis_query(
        layer_url,
        fields="*",
        bbox=bbox
    )

    if layer.empty:
        return layer

    layer = layer.to_crs(
        "EPSG:27700"
    )

    ladywood_union = (
        ladywood.geometry.unary_union
    )

    layer = layer[
        layer.geometry.intersects(
            ladywood_union
        )
    ].copy()

    if layer.empty:
        return layer

    layer["geometry"] = (
        layer.geometry
        .intersection(ladywood_union)
    )

    return repair_geometry(layer)


# ============================================================
# BROWNFIELD REGISTER
# ============================================================

def get_brownfield(ladywood):

    bbox = get_bbox(ladywood)

    brownfield = arcgis_query(
        BROWNFIELD_LAYER,
        fields="*",
        bbox=bbox
    )

    if brownfield.empty:
        return brownfield

    brownfield = brownfield.to_crs(
        "EPSG:27700"
    )

    ladywood_union = (
        ladywood.geometry.unary_union
    )

    brownfield = brownfield[
        brownfield.geometry.intersects(
            ladywood_union
        )
    ].copy()

    if brownfield.empty:
        return brownfield

    brownfield["geometry"] = (
        brownfield.geometry
        .intersection(ladywood_union)
    )

    return repair_geometry(
        brownfield
    )


# ============================================================
# OFFICIAL ROAD CLASSIFICATION
# ============================================================

def get_roads(ladywood):

    bbox = get_bbox(ladywood)

    roads = arcgis_query(
        ROAD_CLASS_LAYER,
        fields="*",
        bbox=bbox
    )

    if roads.empty:
        return roads

    roads = roads.to_crs(
        "EPSG:27700"
    )

    ladywood_union = (
        ladywood.geometry.unary_union
    )

    roads = roads[
        roads.geometry.intersects(
            ladywood_union
        )
    ].copy()

    if roads.empty:
        return roads

    roads["geometry"] = (
        roads.geometry
        .intersection(ladywood_union)
    )

    roads = repair_geometry(
        roads
    )

    # Official Birmingham field.
    if "BRUM_CLASS" not in roads.columns:
        raise RuntimeError(
            "The official road classification field "
            "BRUM_CLASS was not returned."
        )

    if "BRUM_ROAD_" not in roads.columns:
        roads["BRUM_ROAD_"] = "Unnamed road"

    roads["BRUM_CLASS"] = (
        roads["BRUM_CLASS"]
        .fillna("Other")
        .astype(str)
    )

    roads["BRUM_ROAD_"] = (
        roads["BRUM_ROAD_"]
        .fillna("Unnamed road")
        .astype(str)
    )

    return roads


# ============================================================
# NORMALISATION
# ============================================================

def normalise(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(
            np.nan,
            index=series.index
        )

    if maximum == minimum:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def classify(value):

    if pd.isna(value):
        return "NO DATA"

    if value >= 0.67:
        return "HIGH"

    if value >= 0.34:
        return "MODERATE"

    return "LOW"


# ============================================================
# ENVIRONMENTAL CALCULATIONS
# ============================================================

def calculate_results(
    ladywood,
    lsoa,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
):

    data = lsoa.copy()

    data = data.to_crs(
        "EPSG:27700"
    )

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    data["Area_m2"] = (
        data.geometry.area
    )

    data = data[
        data["Area_m2"] > 0
    ].copy()

    # --------------------------------------------------------
    # FIND CRVA FIELDS
    # --------------------------------------------------------

    crva_field = None

    for field in [
        "MEAN",
        "AVERAGE_RISK",
        "MEDIAN"
    ]:

        if field in data.columns:
            crva_field = field
            break

    if crva_field is None:
        raise RuntimeError(
            "The CRVA mean/risk field was not found."
        )

    data["CRVA_Mean"] = pd.to_numeric(
        data[crva_field],
        errors="coerce"
    )

    # --------------------------------------------------------
    # FLOOD AREA
    # --------------------------------------------------------

    if not flood_zone_3.empty:

        flood_union = (
            flood_zone_3
            .to_crs("EPSG:27700")
            .geometry
            .unary_union
        )

    else:
        flood_union = None

    if not surface_flood.empty:

        surface_union = (
            surface_flood
            .to_crs("EPSG:27700")
            .geometry
            .unary_union
        )

    else:
        surface_union = None

    flood_zone_values = []
    surface_values = []

    for geom in data.geometry:

        if flood_union is not None:
            flood_area = geom.intersection(
                flood_union
            ).area
        else:
            flood_area = 0.0

        if surface_union is not None:
            surface_area = geom.intersection(
                surface_union
            ).area
        else:
            surface_area = 0.0

        flood_zone_values.append(
            flood_area
        )

        surface_values.append(
            surface_area
        )

    data["Flood_Zone_3_m2"] = (
        flood_zone_values
    )

    data["Surface_Flood_m2"] = (
        surface_values
    )

    data["Flood_Zone_3_pct"] = (
        data["Flood_Zone_3_m2"]
        / data["Area_m2"]
        * 100
    )

    data["Surface_Flood_pct"] = (
        data["Surface_Flood_m2"]
        / data["Area_m2"]
        * 100
    )

    # We use the larger percentage rather than adding
    # overlapping flood layers together.
    data["Flood_Exposure_pct"] = data[
        [
            "Flood_Zone_3_pct",
            "Surface_Flood_pct"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # BROWNFIELD
    # --------------------------------------------------------

    data["Brownfield_m2"] = 0.0
    data["Brownfield_pct"] = 0.0
    data["Brownfield_Sites"] = 0

    if not brownfield.empty:

        brownfield = (
            brownfield
            .to_crs("EPSG:27700")
        )

        brown_union = (
            brownfield
            .geometry
            .unary_union
        )

        data["Brownfield_m2"] = (
            data.geometry.apply(
                lambda geom:
                geom.intersection(
                    brown_union
                ).area
            )
        )

        data["Brownfield_pct"] = (
            data["Brownfield_m2"]
            / data["Area_m2"]
            * 100
        )

        site_counts = []

        for geom in data.geometry:

            count = brownfield[
                brownfield.geometry.intersects(
                    geom
                )
            ].shape[0]

            site_counts.append(
                count
            )

        data["Brownfield_Sites"] = (
            site_counts
        )

    # --------------------------------------------------------
    # TRAFFIC / AIR SCREENING
    # --------------------------------------------------------

    data["A_Road_100m"] = False
    data["B_Road_100m"] = False
    data["Air_Road_Screening"] = "LOW"

    if not roads.empty:

        roads = roads.to_crs(
            "EPSG:27700"
        )

        a_roads = roads[
            roads["BRUM_CLASS"]
            .str.strip()
            .str.lower()
            == "a road"
        ].copy()

        b_roads = roads[
            roads["BRUM_CLASS"]
            .str.strip()
            .str.lower()
            == "b road"
        ].copy()

        if not a_roads.empty:

            a_corridor = (
                a_roads
                .geometry
                .buffer(100)
                .unary_union
            )

        else:
            a_corridor = None

        if not b_roads.empty:

            b_corridor = (
                b_roads
                .geometry
                .buffer(100)
                .unary_union
            )

        else:
            b_corridor = None

        a_flags = []
        b_flags = []

        for geom in data.geometry:

            if a_corridor is not None:
                a_flag = geom.intersects(
                    a_corridor
                )
            else:
                a_flag = False

            if b_corridor is not None:
                b_flag = geom.intersects(
                    b_corridor
                )
            else:
                b_flag = False

            a_flags.append(
                bool(a_flag)
            )

            b_flags.append(
                bool(b_flag)
            )

        data["A_Road_100m"] = a_flags
        data["B_Road_100m"] = b_flags

        data.loc[
            data["A_Road_100m"],
            "Air_Road_Screening"
        ] = "HIGH"

        data.loc[
            (
                ~data["A_Road_100m"]
                & data["B_Road_100m"]
            ),
            "Air_Road_Screening"
        ] = "MODERATE"

    # --------------------------------------------------------
    # NORMALISED INDICATORS
    # --------------------------------------------------------

    data["CRVA_Index"] = normalise(
        data["CRVA_Mean"]
    )

    data["Flood_Index"] = normalise(
        data["Flood_Exposure_pct"]
    )

    data["Brownfield_Index"] = normalise(
        data["Brownfield_pct"]
    )

    # --------------------------------------------------------
    # OVERALL ENVIRONMENTAL SCREENING
    # --------------------------------------------------------

    data["Environmental_Index"] = (
        data["CRVA_Index"]
        + data["Flood_Index"]
        + data["Brownfield_Index"]
    ) / 3

    data["Environmental_Percent"] = (
        data["Environmental_Index"]
        * 100
    )

    data["Risk_Class"] = (
        data["Environmental_Index"]
        .apply(classify)
    )

    # --------------------------------------------------------
    # LSOA NAME
    # --------------------------------------------------------

    name_field = None

    for field in [
        "LSOA21NM",
        "LSOA_NAME",
        "LSOA21_NAME"
    ]:

        if field in data.columns:
            name_field = field
            break

    if name_field is None:
        data["LSOA_Name"] = (
            data.index.astype(str)
        )
    else:
        data["LSOA_Name"] = (
            data[name_field]
            .astype(str)
        )

    # --------------------------------------------------------
    # LSOA CODE
    # --------------------------------------------------------

    code_field = None

    for field in [
        "LSOA21CD",
        "LSOA_CODE",
        "LSOA21_CODE"
    ]:

        if field in data.columns:
            code_field = field
            break

    if code_field is None:
        data["LSOA_Code"] = (
            data.index.astype(str)
        )
    else:
        data["LSOA_Code"] = (
            data[code_field]
            .astype(str)
        )

    # --------------------------------------------------------
    # CORRECT MAP COORDINATES
    # --------------------------------------------------------

    centroids = data[
        [
            "LSOA_Code",
            "geometry"
        ]
    ].copy()

    centroids["geometry"] = (
        centroids
        .geometry
        .centroid
    )

    centroids = (
        centroids
        .set_crs("EPSG:27700")
        .to_crs("EPSG:4326")
    )

    coordinate_lookup = (
        centroids
        .set_index("LSOA_Code")
    )

    data["Latitude"] = data[
        "LSOA_Code"
    ].map(
        coordinate_lookup.geometry.y
    )

    data["Longitude"] = data[
        "LSOA_Code"
    ].map(
        coordinate_lookup.geometry.x
    )

    return data


# ============================================================
# DEFRA AIR QUALITY
# ============================================================

def find_date_column(columns):

    preferred = [
        "Date Time",
        "DateTime",
        "Datetime",
        "date_time",
        "Date",
        "date"
    ]

    for column in preferred:

        if column in columns:
            return column

    for column in columns:

        text = str(
            column
        ).lower()

        if (
            "date" in text
            or "time" in text
        ):
            return column

    return None


def find_pollutant_column(
    columns,
    pollutant
):

    matches = []

    for column in columns:

        text = (
            str(column)
            .lower()
            .replace("₂", "2")
            .replace("₅", "5")
            .replace(" ", "")
            .replace("_", "")
        )

        if pollutant == "NO2":

            if (
                "no2" in text
                or "nitrogendioxide" in text
            ):
                matches.append(
                    column
                )

        elif pollutant == "PM25":

            if (
                "pm25" in text
                or "pm2.5" in text
                or "particulatematter25" in text
            ):
                matches.append(
                    column
                )

    if not matches:
        return None

    # Prefer concentration-like columns.
    for column in matches:

        text = str(
            column
        ).lower()

        if (
            "concentration" in text
            or "ug" in text
            or "µg" in text
            or "mass" in text
        ):
            return column

    return matches[0]


@st.cache_data(ttl=1800)
def load_air_quality():

    all_records = []

    for year in AIR_YEARS:

        url = (
            f"{DEFRA_BASE}"
            f"{DEFRA_SITE}_{year}.csv"
        )

        try:

            response = requests.get(
                url,
                timeout=90
            )

            response.raise_for_status()

            df = pd.read_csv(
                io.BytesIO(
                    response.content
                ),
                encoding="latin1",
                low_memory=False
            )

            if df.empty:
                continue

            date_column = find_date_column(
                df.columns
            )

            if date_column is None:
                continue

            dates = pd.to_datetime(
                df[date_column],
                errors="coerce"
            )

            for pollutant in [
                "NO2",
                "PM25"
            ]:

                pollutant_column = (
                    find_pollutant_column(
                        df.columns,
                        pollutant
                    )
                )

                if pollutant_column is None:
                    continue

                values = pd.to_numeric(
                    df[pollutant_column],
                    errors="coerce"
                )

                temp = pd.DataFrame({
                    "Date": dates,
                    "Concentration": values,
                    "Pollutant": pollutant
                })

                temp = temp.dropna(
                    subset=[
                        "Date",
                        "Concentration"
                    ]
                )

                if not temp.empty:
                    all_records.append(
                        temp
                    )

        except Exception:
            # Missing data are left missing.
            # They are never replaced with zero.
            continue

    if not all_records:

        return pd.DataFrame(
            columns=[
                "Date",
                "Concentration",
                "Pollutant"
            ]
        )

    air = pd.concat(
        all_records,
        ignore_index=True
    )

    air["Year"] = (
        air["Date"]
        .dt.year
    )

    air["Month"] = (
        air["Date"]
        .dt.to_period("M")
        .astype(str)
    )

    return air.sort_values(
        "Date"
    )


# ============================================================
# MAP
# ============================================================

def build_map(
    ladywood,
    results,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
):

    boundary = ladywood.to_crs(
        "EPSG:4326"
    )

    centre = (
        boundary
        .geometry
        .unary_union
        .centroid
    )

    m = folium.Map(
        location=[
            centre.y,
            centre.x
        ],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    # --------------------------------------------------------
    # LADYWOOD BOUNDARY
    # --------------------------------------------------------

    folium.GeoJson(
        boundary.to_json(),
        name="Ladywood boundary",
        style_function=lambda feature: {
            "fillColor": "#ffffff",
            "fillOpacity": 0.02,
            "color": "#000000",
            "weight": 4
        }
    ).add_to(m)

    # --------------------------------------------------------
    # ENVIRONMENTAL RISK AREAS
    # --------------------------------------------------------

    risk_colours = {
        "HIGH": "#d7191c",
        "MODERATE": "#fdae61",
        "LOW": "#1a9641",
        "NO DATA": "#808080"
    }

    risk_map = results.to_crs(
        "EPSG:4326"
    ).copy()

    map_fields = [
        "LSOA_Code",
        "LSOA_Name",
        "Risk_Class",
        "Environmental_Percent",
        "CRVA_Mean",
        "Flood_Exposure_pct",
        "Brownfield_pct",
        "Air_Road_Screening"
    ]

    def style_function(feature):

        risk = feature[
            "properties"
        ].get(
            "Risk_Class",
            "NO DATA"
        )

        return {
            "fillColor": risk_colours.get(
                risk,
                "#808080"
            ),
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.38
        }

    folium.GeoJson(
        risk_map.to_json(),
        name="Ladywood environmental screening",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=map_fields,
            aliases=[
                "Area:",
                "LSOA code:",
                "Risk:",
                "Environmental score:",
                "CRVA mean:",
                "Flood exposure:",
                "Brownfield:",
                "Road-air screening:"
            ],
            localize=True
        )
    ).add_to(m)

    # --------------------------------------------------------
    # FLOOD ZONE 3
    # --------------------------------------------------------

    if not flood_zone_3.empty:

        flood_map = flood_zone_3.to_crs(
            "EPSG:4326"
        )

        folium.GeoJson(
            flood_map.to_json(),
            name="Flood Zone 3",
            style_function=lambda feature: {
                "fillColor": "#3182bd",
                "color": "#08519c",
                "weight": 1,
                "fillOpacity": 0.25
            }
        ).add_to(m)

    # --------------------------------------------------------
    # SURFACE FLOODING
    # --------------------------------------------------------

    if not surface_flood.empty:

        surface_map = surface_flood.to_crs(
            "EPSG:4326"
        )

        folium.GeoJson(
            surface_map.to_json(),
            name="Surface flood evidence",
            style_function=lambda feature: {
                "fillColor": "#6baed6",
                "color": "#2171b5",
                "weight": 1,
                "fillOpacity": 0.20
            }
        ).add_to(m)

    # --------------------------------------------------------
    # BROWNFIELD
    # --------------------------------------------------------

    if not brownfield.empty:

        brown_map = brownfield.to_crs(
            "EPSG:4326"
        )

        folium.GeoJson(
            brown_map.to_json(),
            name="Brownfield Register 2025",
            style_function=lambda feature: {
                "fillColor": "#8c510a",
                "color": "#5d3a00",
                "weight": 1,
                "fillOpacity": 0.45
            }
        ).add_to(m)

    # --------------------------------------------------------
    # ROAD CORRIDORS
    # --------------------------------------------------------

    if not roads.empty:

        road_map = roads.to_crs(
            "EPSG:4326"
        )

        road_group = folium.FeatureGroup(
            name="Major road corridors"
        )

        for _, road in road_map.iterrows():

            road_type = str(
                road.get(
                    "BRUM_CLASS",
                    "Other"
                )
            )

            road_name = str(
                road.get(
                    "BRUM_ROAD_",
                    "Unnamed road"
                )
            )

            if road_type == "A Road":

                line_colour = "#8b0000"
                line_width = 5

            elif road_type == "B Road":

                line_colour = "#e67e22"
                line_width = 4

            else:

                line_colour = "#777777"
                line_width = 2

            folium.GeoJson(
                road.geometry.__geo_interface__,
                style_function=lambda feature,
                    colour=line_colour,
                    width=line_width: {
                        "color": colour,
                        "weight": width,
                        "opacity": 0.85
                    },
                tooltip=(
                    f"{road_name} — "
                    f"{road_type}"
                )
            ).add_to(
                road_group
            )

        road_group.add_to(m)

    # --------------------------------------------------------
    # RISK MARKERS
    # --------------------------------------------------------

    marker_group = folium.FeatureGroup(
        name="Risk markers"
    )

    for _, row in results.iterrows():

        risk = row["Risk_Class"]

        if risk == "HIGH":
            marker_colour = "red"

        elif risk == "MODERATE":
            marker_colour = "orange"

        elif risk == "LOW":
            marker_colour = "green"

        else:
            marker_colour = "gray"

        popup_text = f"""
        <b>{row['LSOA_Name']}</b><br><br>

        <b>Environmental screening:</b>
        {risk}<br>

        <b>Environmental score:</b>
        {row['Environmental_Percent']:.1f}%<br>

        <b>CRVA mean:</b>
        {row['CRVA_Mean']:.2f}<br>

        <b>Flood exposure:</b>
        {row['Flood_Exposure_pct']:.1f}%<br>

        <b>Brownfield:</b>
        {row['Brownfield_pct']:.1f}%<br>

        <b>Road-air screening:</b>
        {row['Air_Road_Screening']}
        """

        folium.Marker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            popup=folium.Popup(
                popup_text,
                max_width=350
            ),
            tooltip=(
                f"{risk} — "
                f"{row['LSOA_Name']}"
            ),
            icon=folium.Icon(
                color=marker_colour,
                icon="info-sign"
            )
        ).add_to(
            marker_group
        )

    marker_group.add_to(m)

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading official Ladywood spatial data..."
    ):

        ladywood = get_ladywood()

        lsoa = get_lsoa_crva(
            ladywood
        )

        flood_zone_3 = get_spatial_layer(
            FLOOD_ZONE_3_LAYER,
            ladywood
        )

        surface_flood = get_spatial_layer(
            SURFACE_FLOOD_LAYER,
            ladywood
        )

        brownfield = get_brownfield(
            ladywood
        )

        roads = get_roads(
            ladywood
        )

except Exception as error:

    st.error(
        "The Ladywood spatial data could not be loaded."
    )

    st.error(
        str(error)
    )

    st.stop()


# ============================================================
# CALCULATE RESULTS
# ============================================================

results = calculate_results(
    ladywood,
    lsoa,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
)


# ============================================================
# AIR DATA
# ============================================================

with st.spinner(
    "Loading DEFRA Birmingham Ladywood air-quality data..."
):

    air = load_air_quality()


# ============================================================
# SUMMARY
# ============================================================

st.subheader(
    "Ladywood screening overview"
)

high_count = int(
    (
        results["Risk_Class"]
        == "HIGH"
    ).sum()
)

moderate_count = int(
    (
        results["Risk_Class"]
        == "MODERATE"
    ).sum()
)

low_count = int(
    (
        results["Risk_Class"]
        == "LOW"
    ).sum()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Ladywood areas assessed",
    len(results)
)

col2.metric(
    "HIGH",
    high_count
)

col3.metric(
    "MODERATE",
    moderate_count
)

col4.metric(
    "LOW",
    low_count
)


# ============================================================
# MAP
# ============================================================

st.subheader(
    "Ladywood environmental risk map"
)

st.write(
    "The map combines Ladywood spatial evidence. "
    "Use the layer control to switch flood, brownfield and "
    "road evidence on or off."
)

ladywood_map = build_map(
    ladywood,
    results,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
)

st_folium(
    ladywood_map,
    width=None,
    height=700,
    returned_objects=[]
)

st.markdown(
    """
**Map key**

🔴 HIGH

🟠 MODERATE

🟢 LOW

🔴 Dark red road lines = A Roads

🟠 Orange road lines = B Roads

🔵 Blue = flood evidence

🟤 Brown = Brownfield Register

The coloured markers are environmental screening locations,
not individual air-monitoring stations.
"""
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

st.subheader(
    "Environmental screening classification"
)

risk_chart = (
    results["Risk_Class"]
    .value_counts()
    .reindex(
        [
            "HIGH",
            "MODERATE",
            "LOW"
        ],
        fill_value=0
    )
    .reset_index()
)

risk_chart.columns = [
    "Risk",
    "Number of areas"
]

fig = px.bar(
    risk_chart,
    x="Risk",
    y="Number of areas",
    text="Number of areas"
)

fig.update_traces(
    textposition="outside"
)

fig.update_layout(
    yaxis=dict(
        dtick=1
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# CRVA
# ============================================================

st.subheader(
    "Climate Risk and Vulnerability Assessment"
)

crva_chart = (
    results[
        [
            "LSOA_Name",
            "CRVA_Mean"
        ]
    ]
    .dropna()
    .sort_values(
        "CRVA_Mean",
        ascending=False
    )
)

fig_crva = px.bar(
    crva_chart,
    x="LSOA_Name",
    y="CRVA_Mean",
    text="CRVA_Mean"
)

fig_crva.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

fig_crva.update_layout(
    xaxis_title="Ladywood area",
    yaxis_title="CRVA mean"
)

st.plotly_chart(
    fig_crva,
    use_container_width=True
)


# ============================================================
# FLOOD
# ============================================================

st.subheader(
    "Flood exposure by Ladywood area"
)

flood_chart = (
    results[
        [
            "LSOA_Name",
            "Flood_Exposure_pct"
        ]
    ]
    .sort_values(
        "Flood_Exposure_pct",
        ascending=False
    )
)

fig_flood = px.bar(
    flood_chart,
    x="LSOA_Name",
    y="Flood_Exposure_pct",
    text="Flood_Exposure_pct"
)

fig_flood.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig_flood.update_layout(
    xaxis_title="Ladywood area",
    yaxis_title="Flood exposure (%)"
)

st.plotly_chart(
    fig_flood,
    use_container_width=True
)


# ============================================================
# BROWNFIELD
# ============================================================

st.subheader(
    "Brownfield exposure by Ladywood area"
)

brown_chart = (
    results[
        [
            "LSOA_Name",
            "Brownfield_pct",
            "Brownfield_Sites"
        ]
    ]
    .sort_values(
        "Brownfield_pct",
        ascending=False
    )
)

fig_brown = px.bar(
    brown_chart,
    x="LSOA_Name",
    y="Brownfield_pct",
    text="Brownfield_pct"
)

fig_brown.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig_brown.update_layout(
    xaxis_title="Ladywood area",
    yaxis_title="Brownfield coverage (%)"
)

st.plotly_chart(
    fig_brown,
    use_container_width=True
)


# ============================================================
# TRAFFIC / AIR SCREENING
# ============================================================

st.subheader(
    "Traffic-related air-pollution screening"
)

st.write(
    "The dashboard checks which Ladywood areas intersect a "
    "100 m corridor around official classified A and B roads. "
    "This is a transparent spatial screening method; it is not "
    "presented as a legal air-quality limit or as measured NO₂."
)

traffic_table = results[
    [
        "LSOA_Name",
        "Air_Road_Screening",
        "A_Road_100m",
        "B_Road_100m"
    ]
].copy()

traffic_table.columns = [
    "Ladywood area",
    "Road-air screening",
    "Within 100 m of A Road",
    "Within 100 m of B Road"
]

st.dataframe(
    traffic_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MEASURED DEFRA AIR QUALITY
# ============================================================

st.subheader(
    "Measured air quality — Birmingham Ladywood"
)

st.caption(
    "DEFRA measurements from the Birmingham Ladywood monitoring "
    "site. These measurements are shown as Ladywood-wide monitoring "
    "evidence and are not artificially assigned to individual LSOAs."
)

if air.empty:

    st.warning(
        "DEFRA Ladywood air-quality data could not be read. "
        "No missing values have been replaced with zero."
    )

else:

    pollutants = (
        air["Pollutant"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_pollutant = st.selectbox(
        "Select pollutant",
        pollutants
    )

    selected_air = air[
        air["Pollutant"]
        == selected_pollutant
    ].copy()

    monthly_air = (
        selected_air
        .groupby(
            "Month",
            as_index=False
        )["Concentration"]
        .mean()
    )

    monthly_air["Date"] = pd.to_datetime(
        monthly_air["Month"]
    )

    fig_air = px.line(
        monthly_air,
        x="Date",
        y="Concentration",
        markers=True
    )

    fig_air.update_layout(
        xaxis_title="Month",
        yaxis_title="Mean measured concentration"
    )

    st.plotly_chart(
        fig_air,
        use_container_width=True
    )

    annual_air = (
        selected_air
        .groupby(
            "Year",
            as_index=False
        )["Concentration"]
        .mean()
    )

    fig_annual = px.bar(
        annual_air,
        x="Year",
        y="Concentration",
        text="Concentration"
    )

    fig_annual.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    fig_annual.update_layout(
        xaxis_title="Year",
        yaxis_title="Annual mean measured concentration"
    )

    st.plotly_chart(
        fig_annual,
        use_container_width=True
    )


# ============================================================
# ROAD LIST
# ============================================================

st.subheader(
    "Classified road corridors within Ladywood"
)

if roads.empty:

    st.info(
        "No classified road features were returned."
    )

else:

    road_table = roads[
        [
            "BRUM_ROAD_",
            "BRUM_CLASS"
        ]
    ].copy()

    road_table.columns = [
        "Road",
        "Birmingham road classification"
    ]

    road_table = (
        road_table
        .drop_duplicates()
        .sort_values(
            [
                "Birmingham road classification",
                "Road"
            ]
        )
    )

    st.dataframe(
        road_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PRIORITY AREAS
# ============================================================

st.subheader(
    "Priority Ladywood areas for further investigation"
)

priority = (
    results[
        [
            "LSOA_Name",
            "Risk_Class",
            "Environmental_Percent",
            "CRVA_Mean",
            "Flood_Exposure_pct",
            "Brownfield_pct",
            "Air_Road_Screening"
        ]
    ]
    .sort_values(
        "Environmental_Percent",
        ascending=False
    )
    .head(10)
)

priority.columns = [
    "Ladywood area",
    "Screening",
    "Environmental score (%)",
    "CRVA mean",
    "Flood exposure (%)",
    "Brownfield (%)",
    "Road-air screening"
]

st.dataframe(
    priority,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# METHODOLOGY
# ============================================================

st.subheader(
    "How the screening calculation works"
)

st.markdown(
    """
### 1. Ladywood boundary

The official Birmingham City Council Ladywood ward boundary is
used first.

Only spatial features that intersect Ladywood are retained.

### 2. CRVA

The official Birmingham 2025 CRVA data are used to obtain the
CRVA value for each 2021 LSOA intersecting Ladywood.

### 3. Flooding

The dashboard measures the percentage of each Ladywood LSOA
affected by:

- Flood Zone 3
- Surface-water flood evidence

The larger of the two percentages is used as the combined
flood-exposure indicator so that overlapping flood layers are
not simply added together.

### 4. Brownfield

The official Birmingham 2025 Brownfield Register is intersected
with each Ladywood LSOA.

The calculation is:

`Brownfield percentage = brownfield area / LSOA area × 100`

### 5. Traffic and air-pollution screening

Official Birmingham road classification data identify A Roads
and B Roads.

A 100 m spatial corridor is used as a project screening distance.

If an LSOA intersects an A Road corridor:

`HIGH road-air screening`

If it does not intersect an A Road corridor but intersects a
B Road corridor:

`MODERATE road-air screening`

Otherwise:

`LOW road-air screening`

This does NOT claim that the LSOA has a measured NO₂ concentration.
It identifies areas where traffic infrastructure provides a
reasonable reason for further air-quality investigation.

### 6. Measured Ladywood air quality

DEFRA's Birmingham Ladywood monitoring station provides measured
air-quality data, including NO₂ and PM₂.₅.

The measured data are shown as Ladywood-wide monitoring evidence.

The dashboard does not pretend that the single monitoring station
measures pollution separately in every LSOA.

### 7. Overall environmental screening

Three spatial indicators are normalised within Ladywood:

- CRVA
- Flood exposure
- Brownfield exposure

They receive equal weighting:

`Environmental Index = (CRVA Index + Flood Index + Brownfield Index) / 3`

Then:

`Environmental Score = Environmental Index × 100`

Classification:

`67% or higher = HIGH`

`34% to less than 67% = MODERATE`

`Below 34% = LOW`

These are project screening categories created to make the evidence
easy to compare. They are NOT claimed to be Birmingham City
Council's official risk thresholds.
"""
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.subheader(
    "Ladywood environmental results"
)

download_columns = [
    "LSOA_Code",
    "LSOA_Name",
    "CRVA_Mean",
    "Flood_Zone_3_pct",
    "Surface_Flood_pct",
    "Flood_Exposure_pct",
    "Brownfield_pct",
    "Brownfield_Sites",
    "A_Road_100m",
    "B_Road_100m",
    "Air_Road_Screening",
    "Environmental_Index",
    "Environmental_Percent",
    "Risk_Class",
    "Latitude",
    "Longitude"
]

download_columns = [
    column
    for column in download_columns
    if column in results.columns
]

csv_data = (
    results[
        download_columns
    ]
    .to_csv(
        index=False
    )
    .encode("utf-8")
)

st.download_button(
    label="Download Ladywood environmental results",
    data=csv_data,
    file_name="Ladywood_Environmental_Results.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Study area: Ladywood, Birmingham | "
    "Spatial evidence: Birmingham City Council | "
    "Air monitoring: DEFRA UK-AIR Birmingham Ladywood"
)
