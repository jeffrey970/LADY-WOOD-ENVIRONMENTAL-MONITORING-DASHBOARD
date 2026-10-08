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
    "Environmental screening of Ladywood using Birmingham City Council "
    "spatial data and DEFRA Birmingham Ladywood monitoring data."
)


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

CRVA_SERVICE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = f"{CRVA_SERVICE}/14"
CRVA_LSOA_LAYER = f"{CRVA_SERVICE}/15"
LSOA_BOUNDARY_LAYER = f"{CRVA_SERVICE}/17"

FLOOD_ZONE_3_LAYER = f"{CRVA_SERVICE}/109"
SURFACE_FLOOD_LAYER = f"{CRVA_SERVICE}/1546"

# Official Birmingham CRVA pollution layers.
# These are raster layers and are displayed on the map as
# contextual pollution evidence. They are NOT treated as
# individual monitoring stations.
NO2_RASTER_LAYER = f"{CRVA_SERVICE}/8"
PM25_RASTER_LAYER = f"{CRVA_SERVICE}/9"

# Birmingham highways
ROAD_CLASS_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_Highways_open/MapServer/26"
)

ROAD_NAME_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "Internet_HLC/MapServer/6"
)

# Correct 2025 Birmingham Brownfield Register
BROWNFIELD_LAYER = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer/38"
)

# DEFRA Birmingham Ladywood monitoring station
DEFRA_BASE = (
    "https://uk-air.defra.gov.uk/datastore/data_files/site_data/"
)

DEFRA_SITE = "BMLD"

YEARS = list(range(2019, 2027))


# ============================================================
# GENERAL HELPERS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def arcgis_query(
    layer_url,
    where="1=1",
    out_fields="*",
    geometry=None,
    geometry_type="esriGeometryEnvelope",
    spatial_rel="esriSpatialRelIntersects",
):
    """
    Query an ArcGIS REST Feature Layer and return a GeoDataFrame.

    All spatial processing is eventually done in British National Grid
    (EPSG:27700), which is appropriate for area and distance calculations
    in metres.
    """

    params = {
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    if geometry is not None:
        params["geometry"] = geometry
        params["geometryType"] = geometry_type
        params["inSR"] = "27700"
        params["spatialRel"] = spatial_rel

    response = requests.get(
        f"{layer_url}/query",
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    if "features" not in data:
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


def repair_geometry(gdf):
    """Repair invalid geometries safely."""

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


def bbox_string(gdf):
    """
    Return an ArcGIS geometry envelope in EPSG:27700.
    """

    bounds = gdf.total_bounds

    xmin, ymin, xmax, ymax = bounds

    return f"{xmin},{ymin},{xmax},{ymax}"


def normalise(series):
    """
    Min-max normalisation.

    If every value is identical, return zero rather than inventing
    variation.
    """

    values = pd.to_numeric(series, errors="coerce")

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

    return (values - minimum) / (maximum - minimum)


def classify(score):
    """
    Project screening classification.

    This is NOT a Birmingham City Council official risk class.
    It is a transparent screening classification for this project.
    """

    if pd.isna(score):
        return "NO DATA"

    if score >= 0.67:
        return "HIGH"

    if score >= 0.34:
        return "MODERATE"

    return "LOW"


# ============================================================
# LADYWOOD BOUNDARY
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_ladywood():

    wards = arcgis_query(
        WARD_LAYER,
        where="1=1",
        out_fields="*"
    )

    if wards.empty:
        raise RuntimeError(
            "Birmingham ward boundary data could not be loaded."
        )

    wards = repair_geometry(wards)

    name_columns = [
        column for column in wards.columns
        if "ward" in column.lower()
        or "name" in column.lower()
    ]

    ladywood = None

    for column in name_columns:
        matches = wards[
            wards[column]
            .astype(str)
            .str.contains(
                "Ladywood",
                case=False,
                na=False
            )
        ]

        if not matches.empty:
            ladywood = matches.copy()
            break

    if ladywood is None or ladywood.empty:
        raise RuntimeError(
            "Ladywood ward could not be identified in the official "
            "Birmingham boundary data."
        )

    ladywood = ladywood.to_crs("EPSG:27700")

    ladywood["geometry"] = ladywood.geometry.buffer(0)

    return ladywood


# ============================================================
# LSOA + CRVA DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_lsoa_crva(ladywood):

    geometry = bbox_string(ladywood)

    crva = arcgis_query(
        CRVA_LSOA_LAYER,
        where="1=1",
        out_fields="*",
        geometry=geometry
    )

    if crva.empty:
        raise RuntimeError(
            "CRVA LSOA data could not be loaded."
        )

    crva = repair_geometry(crva)
    crva = crva.to_crs("EPSG:27700")

    # Keep only LSOAs which actually intersect Ladywood.
    ladywood_union = ladywood.geometry.unary_union

    crva = crva[
        crva.geometry.intersects(ladywood_union)
    ].copy()

    # Clip to Ladywood.
    crva["geometry"] = crva.geometry.intersection(
        ladywood_union
    )

    crva = repair_geometry(crva)

    return crva


# ============================================================
# FLOOD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_flood_layer(layer_url, ladywood):

    geometry = bbox_string(ladywood)

    flood = arcgis_query(
        layer_url,
        where="1=1",
        out_fields="*",
        geometry=geometry
    )

    if flood.empty:
        return flood

    flood = repair_geometry(flood)
    flood = flood.to_crs("EPSG:27700")

    ladywood_union = ladywood.geometry.unary_union

    flood = flood[
        flood.geometry.intersects(ladywood_union)
    ].copy()

    flood["geometry"] = flood.geometry.intersection(
        ladywood_union
    )

    return repair_geometry(flood)


# ============================================================
# BROWNFIELD REGISTER
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_brownfield(ladywood):

    geometry = bbox_string(ladywood)

    brownfield = arcgis_query(
        BROWNFIELD_LAYER,
        where="1=1",
        out_fields="*",
        geometry=geometry
    )

    if brownfield.empty:
        return brownfield

    brownfield = repair_geometry(brownfield)
    brownfield = brownfield.to_crs("EPSG:27700")

    ladywood_union = ladywood.geometry.unary_union

    brownfield = brownfield[
        brownfield.geometry.intersects(ladywood_union)
    ].copy()

    brownfield["geometry"] = brownfield.geometry.intersection(
        ladywood_union
    )

    return repair_geometry(brownfield)


# ============================================================
# ROAD DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_roads(ladywood):

    geometry = bbox_string(ladywood)

    # --------------------------------------------------------
    # A/B road classification
    # --------------------------------------------------------

    road_class = arcgis_query(
        ROAD_CLASS_LAYER,
        where="1=1",
        out_fields="OBJECTID,BRUM_CLASS,BRUM_ROAD_",
        geometry=geometry
    )

    # --------------------------------------------------------
    # Road names
    # --------------------------------------------------------

    road_names = arcgis_query(
        ROAD_NAME_LAYER,
        where="1=1",
        out_fields="OBJECTID,ROAD_NAME,LEGEND",
        geometry=geometry
    )

    if road_class.empty:
        return gpd.GeoDataFrame(
            geometry=[],
            crs="EPSG:27700"
        )

    road_class = repair_geometry(
        road_class.to_crs("EPSG:27700")
    )

    road_class = road_class[
        road_class.geometry.intersects(
            ladywood.geometry.unary_union
        )
    ].copy()

    road_class["geometry"] = road_class.geometry.intersection(
        ladywood.geometry.unary_union
    )

    road_class = repair_geometry(road_class)

    # --------------------------------------------------------
    # Join road names spatially.
    #
    # The classification layer and current-road layer are
    # maintained separately by Birmingham City Council.
    # We find the nearest named road to each classified road.
    # --------------------------------------------------------

    if not road_names.empty:

        road_names = repair_geometry(
            road_names.to_crs("EPSG:27700")
        )

        road_names = road_names[
            road_names.geometry.intersects(
                ladywood.geometry.unary_union
            )
        ].copy()

        if not road_names.empty:

            road_names["geometry"] = road_names.geometry.intersection(
                ladywood.geometry.unary_union
            )

            road_names = repair_geometry(road_names)

            # Keep useful road names only.
            road_names["ROAD_NAME"] = (
                road_names["ROAD_NAME"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            road_names = road_names[
                road_names["ROAD_NAME"] != ""
            ].copy()

            if not road_names.empty:

                try:
                    road_class = gpd.sjoin_nearest(
                        road_class,
                        road_names[
                            ["ROAD_NAME", "geometry"]
                        ],
                        how="left",
                        distance_col="Road_Name_Distance_m"
                    )
                except Exception:
                    road_class["ROAD_NAME"] = "Unnamed road"

            else:
                road_class["ROAD_NAME"] = "Unnamed road"

        else:
            road_class["ROAD_NAME"] = "Unnamed road"

    else:
        road_class["ROAD_NAME"] = "Unnamed road"

    # --------------------------------------------------------
    # Clean classification
    # --------------------------------------------------------

    road_class["BRUM_CLASS"] = (
        road_class["BRUM_CLASS"]
        .fillna("Other")
        .astype(str)
    )

    road_class["Road_Type"] = road_class["BRUM_CLASS"].replace(
        {
            "A Road": "A Road",
            "B Road": "B Road",
            "Classified Un Numbered": "Classified road",
        }
    )

    # Remove duplicate geometries caused by nearest-name matching.
    road_class = (
        road_class
        .drop_duplicates(
            subset=[
                "ROAD_NAME",
                "BRUM_CLASS",
                "geometry"
            ]
        )
        .reset_index(drop=True)
    )

    return road_class


# ============================================================
# AIR QUALITY DATA
# ============================================================

def find_date_column(columns):

    priority = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Date Time",
        "date_time",
        "DateTime",
    ]

    for name in priority:
        if name in columns:
            return name

    for column in columns:
        lower = str(column).lower()

        if "date" in lower or "time" in lower:
            return column

    return None


def find_pollutant_column(columns, pollutant):

    pollutant = pollutant.lower()

    candidates = []

    for column in columns:

        text = (
            str(column)
            .lower()
            .replace("₂", "2")
            .replace("₅", "5")
        )

        if pollutant == "no2":
            if (
                "nitrogen dioxide" in text
                or "no2" in text
                or "no_2" in text
            ):
                candidates.append(column)

        elif pollutant == "pm25":
            if (
                "pm2.5" in text
                or "pm2_5" in text
                or "pm25" in text
                or "pm 2.5" in text
                or "fine particulate" in text
            ):
                candidates.append(column)

    # Prefer measured concentration columns.
    for column in candidates:

        text = str(column).lower()

        if (
            "concentration" in text
            or "ug" in text
            or "µg" in text
            or "mass" in text
        ):
            return column

    return candidates[0] if candidates else None


@st.cache_data(ttl=1800, show_spinner=False)
def get_air_quality():

    all_years = []

    for year in YEARS:

        url = (
            f"{DEFRA_BASE}"
            f"{DEFRA_SITE}_{year}.csv"
        )

        try:

            response = requests.get(
                url,
                timeout=60
            )

            response.raise_for_status()

            raw = response.content

            # DEFRA files are generally standard CSVs,
            # but latin-1 prevents failures from special characters.
            df = pd.read_csv(
                io.BytesIO(raw),
                low_memory=False,
                encoding="latin1"
            )

            if df.empty:
                continue

            date_column = find_date_column(
                df.columns
            )

            no2_column = find_pollutant_column(
                df.columns,
                "no2"
            )

            pm25_column = find_pollutant_column(
                df.columns,
                "pm25"
            )

            if date_column is None:
                continue

            dates = pd.to_datetime(
                df[date_column],
                errors="coerce"
            )

            # NO2
            if no2_column is not None:

                no2 = pd.DataFrame({
                    "Date": dates,
                    "Concentration": pd.to_numeric(
                        df[no2_column],
                        errors="coerce"
                    ),
                    "Pollutant": "NO₂",
                })

                no2 = no2.dropna(
                    subset=[
                        "Date",
                        "Concentration"
                    ]
                )

                if not no2.empty:
                    all_years.append(no2)

            # PM2.5
            if pm25_column is not None:

                pm25 = pd.DataFrame({
                    "Date": dates,
                    "Concentration": pd.to_numeric(
                        df[pm25_column],
                        errors="coerce"
                    ),
                    "Pollutant": "PM₂.₅",
                })

                pm25 = pm25.dropna(
                    subset=[
                        "Date",
                        "Concentration"
                    ]
                )

                if not pm25.empty:
                    all_years.append(pm25)

        except Exception:
            # A missing/unreadable year is left missing.
            # It is NEVER replaced with zero.
            continue

    if not all_years:

        return pd.DataFrame(
            columns=[
                "Date",
                "Concentration",
                "Pollutant",
                "Year",
                "Month"
            ]
        )

    air = pd.concat(
        all_years,
        ignore_index=True
    )

    air["Year"] = air["Date"].dt.year
    air["Month"] = air["Date"].dt.to_period("M").astype(str)

    air = air.sort_values("Date")

    return air


# ============================================================
# SPATIAL INDICATOR CALCULATION
# ============================================================

def calculate_indicators(
    ladywood,
    lsoa,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
):

    ladywood_union = ladywood.geometry.unary_union

    # --------------------------------------------------------
    # Start with the LSOAs intersecting Ladywood.
    # --------------------------------------------------------

    data = lsoa.copy()

    data["geometry"] = data.geometry.intersection(
        ladywood_union
    )

    data = repair_geometry(data)

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    data["Area_m2"] = data.geometry.area

    data = data[
        data["Area_m2"] > 0
    ].copy()

    # --------------------------------------------------------
    # CRVA
    # --------------------------------------------------------

    if "MEAN" in data.columns:
        data["CRVA_Mean"] = pd.to_numeric(
            data["MEAN"],
            errors="coerce"
        )

    elif "AVERAGE_RISK" in data.columns:
        data["CRVA_Mean"] = pd.to_numeric(
            data["AVERAGE_RISK"],
            errors="coerce"
        )

    else:
        data["CRVA_Mean"] = np.nan

    # --------------------------------------------------------
    # Flood exposure
    # --------------------------------------------------------

    def intersect_area(layer):

        if layer.empty:
            return pd.Series(
                0.0,
                index=data.index
            )

        layer_union = layer.geometry.unary_union

        return data.geometry.apply(
            lambda geom: (
                geom.intersection(layer_union).area
                if not geom.is_empty
                else 0
            )
        )

    data["Flood_Zone_3_m2"] = intersect_area(
        flood_zone_3
    )

    data["Surface_Flood_m2"] = intersect_area(
        surface_flood
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

    # Use the larger percentage rather than adding both.
    # This prevents double-counting where flood datasets overlap.
    data["Flood_Exposure_pct"] = data[
        [
            "Flood_Zone_3_pct",
            "Surface_Flood_pct"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # Brownfield exposure
    # --------------------------------------------------------

    if brownfield.empty:

        data["Brownfield_m2"] = 0.0
        data["Brownfield_pct"] = 0.0
        data["Brownfield_Sites"] = 0

    else:

        brownfield_union = brownfield.geometry.unary_union

        data["Brownfield_m2"] = data.geometry.apply(
            lambda geom: (
                geom.intersection(
                    brownfield_union
                ).area
                if not geom.is_empty
                else 0
            )
        )

        data["Brownfield_pct"] = (
            data["Brownfield_m2"]
            / data["Area_m2"]
            * 100
        )

        # Count brownfield sites touching each LSOA.
        counts = []

        for geom in data.geometry:

            count = brownfield[
                brownfield.geometry.intersects(
                    geom
                )
            ].shape[0]

            counts.append(count)

        data["Brownfield_Sites"] = counts

    # --------------------------------------------------------
    # ROAD / AIR EXPOSURE
    # --------------------------------------------------------

    if roads.empty:

        data["A_Road_Exposure_m"] = 0.0
        data["B_Road_Exposure_m"] = 0.0
        data["A_Road_100m"] = False
        data["B_Road_100m"] = False
        data["Air_Road_Screening"] = "LOW"

    else:

        a_roads = roads[
            roads["Road_Type"] == "A Road"
        ].copy()

        b_roads = roads[
            roads["Road_Type"] == "B Road"
        ].copy()

        # 100 m screening corridor.
        #
        # This is a transparent project screening distance,
        # NOT a legal air-quality standard.
        a_buffer = (
            a_roads
            .buffer(100)
            .union_all()
            if not a_roads.empty
            else None
        )

        b_buffer = (
            b_roads
            .buffer(100)
            .union_all()
            if not b_roads.empty
            else None
        )

        a_lengths = []
        b_lengths = []
        a_flags = []
        b_flags = []

        for geom in data.geometry:

            if a_buffer is not None:
                a_overlap = geom.intersection(
                    a_buffer
                )

                a_area = a_overlap.area
                a_flag = a_area > 0

            else:
                a_area = 0
                a_flag = False

            if b_buffer is not None:
                b_overlap = geom.intersection(
                    b_buffer
                )

                b_area = b_overlap.area
                b_flag = b_area > 0

            else:
                b_area = 0
                b_flag = False

            a_lengths.append(a_area)
            b_lengths.append(b_area)
            a_flags.append(a_flag)
            b_flags.append(b_flag)

        data["A_Road_Exposure_m"] = a_lengths
        data["B_Road_Exposure_m"] = b_lengths

        data["A_Road_100m"] = a_flags
        data["B_Road_100m"] = b_flags

        def road_air_class(row):

            if row["A_Road_100m"]:
                return "HIGH"

            if row["B_Road_100m"]:
                return "MODERATE"

            return "LOW"

        data["Air_Road_Screening"] = data.apply(
            road_air_class,
            axis=1
        )

    # --------------------------------------------------------
    # NORMALISED SCREENING INDICATORS
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
    # Spatial environmental screening index
    #
    # Equal weighting is deliberately used.
    #
    # It avoids inventing unsupported weights such as 50/30/20.
    # --------------------------------------------------------

    data["Environmental_Index"] = (
        data["CRVA_Index"]
        + data["Flood_Index"]
        + data["Brownfield_Index"]
    ) / 3

    data["Environmental_Percent"] = (
        data["Environmental_Index"] * 100
    )

    data["Risk_Class"] = data[
        "Environmental_Index"
    ].apply(classify)

    # --------------------------------------------------------
    # Centroids
    #
    # IMPORTANT:
    # Calculate centroid in EPSG:27700 first, then convert to
    # latitude/longitude. This prevents the old map-coordinate bug.
    # --------------------------------------------------------

    centroids = data.geometry.centroid

    centroid_gdf = gpd.GeoDataFrame(
        data.drop(columns="geometry"),
        geometry=centroids,
        crs="EPSG:27700"
    ).to_crs("EPSG:4326")

    data["Longitude"] = centroid_gdf.geometry.x
    data["Latitude"] = centroid_gdf.geometry.y

    return data


# ============================================================
# AIR QUALITY MONTHLY SUMMARY
# ============================================================

def monthly_air_summary(air):

    if air.empty:
        return pd.DataFrame()

    summary = (
        air
        .groupby(
            [
                "Month",
                "Pollutant"
            ],
            as_index=False
        )["Concentration"]
        .mean()
    )

    summary["Date"] = pd.to_datetime(
        summary["Month"]
    )

    return summary.sort_values("Date")


# ============================================================
# CREATE LADYWOOD MAP
# ============================================================

def make_map(
    ladywood,
    results,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
):

    centre = ladywood.to_crs(
        "EPSG:4326"
    ).geometry.unary_union.centroid

    m = folium.Map(
        location=[
            centre.y,
            centre.x
        ],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    # --------------------------------------------------------
    # Ladywood boundary
    # --------------------------------------------------------

    folium.GeoJson(
        ladywood.to_crs("EPSG:4326").to_json(),
        name="Ladywood boundary",
        style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "#000000",
            "weight": 4,
            "fillOpacity": 0
        }
    ).add_to(m)

    # --------------------------------------------------------
    # LSOA screening areas
    # --------------------------------------------------------

    risk_colours = {
        "HIGH": "#d7191c",
        "MODERATE": "#fdae61",
        "LOW": "#1a9641",
        "NO DATA": "#808080"
    }

    result_map = results[
        [
            "LSOA21CD",
            "LSOA21NM",
            "Risk_Class",
            "Environmental_Percent",
            "CRVA_Mean",
            "Flood_Exposure_pct",
            "Brownfield_pct",
            "Air_Road_Screening",
            "geometry"
        ]
    ].copy()

    result_map = result_map.to_crs(
        "EPSG:4326"
    )

    def lsoa_style(feature):

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
            "fillOpacity": 0.35
        }

    folium.GeoJson(
        result_map.to_json(),
        name="Environmental screening",
        style_function=lsoa_style,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "LSOA21CD",
                "LSOA21NM",
                "Risk_Class",
                "Environmental_Percent",
                "CRVA_Mean",
                "Flood_Exposure_pct",
                "Brownfield_pct",
                "Air_Road_Screening"
            ],
            aliases=[
                "LSOA:",
                "Area:",
                "Screening:",
                "Environmental score (%):",
                "CRVA mean:",
                "Flood exposure (%):",
                "Brownfield (%):",
                "Road-air screening:"
            ],
            localize=True,
            sticky=False
        )
    ).add_to(m)

    # --------------------------------------------------------
    # Flood Zone 3
    # --------------------------------------------------------

    if not flood_zone_3.empty:

        folium.GeoJson(
            flood_zone_3.to_crs(
                "EPSG:4326"
            ).to_json(),
            name="Flood Zone 3",
            style_function=lambda feature: {
                "fillColor": "#3182bd",
                "color": "#08519c",
                "weight": 1,
                "fillOpacity": 0.30
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Surface flooding
    # --------------------------------------------------------

    if not surface_flood.empty:

        folium.GeoJson(
            surface_flood.to_crs(
                "EPSG:4326"
            ).to_json(),
            name="Surface flood risk",
            style_function=lambda feature: {
                "fillColor": "#6baed6",
                "color": "#2171b5",
                "weight": 1,
                "fillOpacity": 0.20
            }
        ).add_to(m)

    # --------------------------------------------------------
    # Brownfield
    # --------------------------------------------------------

    if not brownfield.empty:

        brownfield_display = brownfield.copy()

        fields = [
            "SiteReference",
            "SiteNameAddress",
            "PlanningStatus",
            "Hectares"
        ]

        available_fields = [
            f for f in fields
            if f in brownfield_display.columns
        ]

        folium.GeoJson(
            brownfield_display.to_crs(
                "EPSG:4326"
            ).to_json(),
            name="Brownfield Register 2025",
            style_function=lambda feature: {
                "fillColor": "#8c510a",
                "color": "#5d3a00",
                "weight": 1,
                "fillOpacity": 0.45
            },
            tooltip=(
                folium.GeoJsonTooltip(
                    fields=available_fields,
                    aliases=[
                        f.replace("_", " ")
                        for f in available_fields
                    ]
                )
                if available_fields
                else None
            )
        ).add_to(m)

    # --------------------------------------------------------
    # A and B roads
    # --------------------------------------------------------

    if not roads.empty:

        road_map = roads.to_crs(
            "EPSG:4326"
        )

        for _, row in road_map.iterrows():

            road_type = row.get(
                "Road_Type",
                "Other"
            )

            road_name = row.get(
                "ROAD_NAME",
                "Unnamed road"
            )

            if road_type == "A Road":
                line_colour = "#7a0000"
                weight = 5

            elif road_type == "B Road":
                line_colour = "#d95f02"
                weight = 4

            else:
                line_colour = "#777777"
                weight = 2

            folium.GeoJson(
                row.geometry.__geo_interface__,
                name="Road network",
                style_function=lambda feature,
                    colour=line_colour,
                    road_weight=weight: {
                        "color": colour,
                        "weight": road_weight,
                        "opacity": 0.8
                    },
                tooltip=(
                    f"{road_name} — {road_type}"
                )
            ).add_to(m)

    # --------------------------------------------------------
    # Risk markers
    # --------------------------------------------------------

    marker_layer = folium.FeatureGroup(
        name="Risk markers"
    )

    for _, row in results.iterrows():

        risk = row["Risk_Class"]

        if risk == "HIGH":
            colour = "red"

        elif risk == "MODERATE":
            colour = "orange"

        elif risk == "LOW":
            colour = "green"

        else:
            colour = "gray"

        popup_html = f"""
        <b>{row.get('LSOA21NM', 'Ladywood area')}</b><br>
        Screening: {risk}<br>
        Environmental score:
        {row['Environmental_Percent']:.1f}%<br>
        CRVA mean:
        {row['CRVA_Mean']:.2f}<br>
        Flood exposure:
        {row['Flood_Exposure_pct']:.1f}%<br>
        Brownfield:
        {row['Brownfield_pct']:.1f}%<br>
        Road-air screening:
        {row['Air_Road_Screening']}
        """

        folium.Marker(
            location=[
                row["Latitude"],
                row["Longitude"]
            ],
            popup=folium.Popup(
                popup_html,
                max_width=320
            ),
            tooltip=f"{risk}: {row.get('LSOA21NM', '')}",
            icon=folium.Icon(
                color=colour,
                icon="info-sign"
            )
        ).add_to(marker_layer)

    marker_layer.add_to(m)

    # --------------------------------------------------------
    # Layer control
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


# ============================================================
# LOAD DATA
# ============================================================

try:

    with st.spinner(
        "Loading official Birmingham spatial data..."
    ):

        ladywood = get_ladywood()

        lsoa = get_lsoa_crva(
            ladywood
        )

        flood_zone_3 = get_flood_layer(
            FLOOD_ZONE_3_LAYER,
            ladywood
        )

        surface_flood = get_flood_layer(
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
        "The Birmingham spatial data could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# CALCULATE RESULTS
# ============================================================

results = calculate_indicators(
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
    "Loading Birmingham Ladywood air-quality data..."
):

    air = get_air_quality()

monthly_air = monthly_air_summary(
    air
)


# ============================================================
# TOP METRICS
# ============================================================

high_count = int(
    (results["Risk_Class"] == "HIGH").sum()
)

moderate_count = int(
    (results["Risk_Class"] == "MODERATE").sum()
)

low_count = int(
    (results["Risk_Class"] == "LOW").sum()
)

st.subheader("Ladywood screening overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Ladywood areas assessed",
    len(results)
)

c2.metric(
    "High screening areas",
    high_count
)

c3.metric(
    "Moderate screening areas",
    moderate_count
)

c4.metric(
    "Low screening areas",
    low_count
)


# ============================================================
# MAP
# ============================================================

st.subheader("Ladywood environmental risk map")

st.write(
    "The map combines the Ladywood boundary, official LSOA areas, "
    "flood exposure, the 2025 Brownfield Register, classified roads "
    "and the calculated environmental screening result."
)

map_object = make_map(
    ladywood,
    results,
    flood_zone_3,
    surface_flood,
    brownfield,
    roads
)

st_folium(
    map_object,
    width=None,
    height=700,
    returned_objects=[]
)


# ============================================================
# MAP EXPLANATION
# ============================================================

st.markdown(
    """
### Map key

🔴 **HIGH** — higher environmental screening result

🟠 **MODERATE** — intermediate environmental screening result

🟢 **LOW** — lower environmental screening result

**Dark red road:** Birmingham classified A Road

**Orange road:** Birmingham classified B Road

**Blue:** Flood Zone 3 / surface-flood evidence

**Brown:** Brownfield Register 2025 site

The coloured risk markers represent the calculated screening
classification of the Ladywood spatial areas. They do not represent
individual air-monitoring stations.
"""
)


# ============================================================
# RISK CLASSIFICATION CHART
# ============================================================

st.subheader("Environmental screening classification")

risk_order = [
    "HIGH",
    "MODERATE",
    "LOW"
]

risk_counts = (
    results["Risk_Class"]
    .value_counts()
    .reindex(
        risk_order,
        fill_value=0
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk_Class",
    "Number_of_Areas"
]

fig_risk = px.bar(
    risk_counts,
    x="Risk_Class",
    y="Number_of_Areas",
    text="Number_of_Areas",
    category_orders={
        "Risk_Class": risk_order
    },
    labels={
        "Risk_Class": "Screening classification",
        "Number_of_Areas": "Number of Ladywood areas"
    }
)

fig_risk.update_traces(
    textposition="outside"
)

fig_risk.update_layout(
    yaxis=dict(
        rangemode="tozero"
    )
)

st.plotly_chart(
    fig_risk,
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
            "LSOA21NM",
            "CRVA_Mean"
        ]
    ]
    .dropna()
    .sort_values(
        "CRVA_Mean",
        ascending=False
    )
)

if not crva_chart.empty:

    fig_crva = px.bar(
        crva_chart,
        x="LSOA21NM",
        y="CRVA_Mean",
        text="CRVA_Mean",
        labels={
            "LSOA21NM": "Ladywood spatial area",
            "CRVA_Mean": "Official CRVA mean"
        }
    )

    fig_crva.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_crva,
        use_container_width=True
    )


# ============================================================
# FLOOD EXPOSURE
# ============================================================

st.subheader(
    "Flood exposure by Ladywood area"
)

flood_chart = (
    results[
        [
            "LSOA21NM",
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
    x="LSOA21NM",
    y="Flood_Exposure_pct",
    text="Flood_Exposure_pct",
    labels={
        "LSOA21NM": "Ladywood spatial area",
        "Flood_Exposure_pct": "Flood exposure (%)"
    }
)

fig_flood.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_flood,
    use_container_width=True
)


# ============================================================
# BROWNFIELD
# ============================================================

st.subheader(
    "Brownfield exposure"
)

brown_chart = (
    results[
        [
            "LSOA21NM",
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
    x="LSOA21NM",
    y="Brownfield_pct",
    text="Brownfield_pct",
    labels={
        "LSOA21NM": "Ladywood spatial area",
        "Brownfield_pct": "Brownfield coverage (%)"
    }
)

fig_brown.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_brown,
    use_container_width=True
)


# ============================================================
# TRAFFIC / AIR EXPOSURE
# ============================================================

st.subheader(
    "Traffic and air-pollution exposure screening"
)

st.write(
    "This section identifies Ladywood areas that intersect a "
    "100 m screening corridor around classified A or B roads. "
    "The distance is used only as a transparent project-screening "
    "assumption; it is not presented as an official air-quality "
    "threshold."
)

air_screening = (
    results[
        [
            "LSOA21NM",
            "Air_Road_Screening",
            "A_Road_100m",
            "B_Road_100m"
        ]
    ]
    .copy()
)

air_screening.columns = [
    "Ladywood area",
    "Road-air screening",
    "Within 100 m of A Road",
    "Within 100 m of B Road"
]

st.dataframe(
    air_screening,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# LADYWOOD AIR QUALITY TREND
# ============================================================

st.subheader(
    "Measured air quality — Birmingham Ladywood monitoring station"
)

st.caption(
    "These measurements come from DEFRA's Birmingham Ladywood "
    "monitoring site. They are Ladywood-wide monitoring evidence "
    "and are not assigned to individual LSOAs."
)

if monthly_air.empty:

    st.warning(
        "DEFRA Ladywood air data could not be read at this time. "
        "No missing values have been replaced with zero."
    )

else:

    pollutants_available = (
        monthly_air["Pollutant"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_pollutant = st.selectbox(
        "Pollutant",
        pollutants_available
    )

    selected_air = monthly_air[
        monthly_air["Pollutant"]
        == selected_pollutant
    ].copy()

    fig_air = px.line(
        selected_air,
        x="Date",
        y="Concentration",
        markers=True,
        labels={
            "Date": "Month",
            "Concentration": "Mean measured concentration"
        }
    )

    st.plotly_chart(
        fig_air,
        use_container_width=True
    )

    annual_air = (
        air[
            air["Pollutant"]
            == selected_pollutant
        ]
        .groupby("Year", as_index=False)
        ["Concentration"]
        .mean()
    )

    if not annual_air.empty:

        st.subheader(
            f"Annual {selected_pollutant} average"
        )

        fig_annual = px.bar(
            annual_air,
            x="Year",
            y="Concentration",
            text="Concentration",
            labels={
                "Year": "Year",
                "Concentration": "Annual mean concentration"
            }
        )

        fig_annual.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_annual,
            use_container_width=True
        )


# ============================================================
# ROAD CORRIDORS
# ============================================================

st.subheader(
    "Major road corridors intersecting Ladywood"
)

if roads.empty:

    st.info(
        "No classified Birmingham road features were returned "
        "inside the Ladywood boundary."
    )

else:

    road_table = (
        roads[
            [
                "ROAD_NAME",
                "Road_Type"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Road_Type",
                "ROAD_NAME"
            ]
        )
    )

    road_table.columns = [
        "Road",
        "Birmingham road classification"
    ]

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
            "LSOA21NM",
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
# HOW THE CALCULATION WORKS
# ============================================================

st.subheader(
    "How the screening calculation works"
)

st.markdown(
    """
The dashboard does **not** invent an official Birmingham risk score.

It creates a transparent project screening index from three
spatial indicators:

**1. CRVA**

The official Birmingham Climate Risk and Vulnerability Assessment
value is used as one indicator of environmental vulnerability.

**2. Flood exposure**

The proportion of each Ladywood spatial area affected by the
available flood evidence is calculated.

Flood Zone 3 and surface-water flooding are considered separately,
then the larger percentage is used so overlapping flood datasets
are not simply added together.

**3. Brownfield exposure**

The proportion of each Ladywood spatial area intersecting the
Birmingham 2025 Brownfield Register is calculated.

The three indicators are normalised within the Ladywood study area
and given equal weighting:

`Environmental Index = (CRVA Index + Flood Index + Brownfield Index) / 3`

The result is converted to a percentage.

The screening classes are:

`67–100% → HIGH`

`34–66% → MODERATE`

`0–33% → LOW`

These thresholds are **project screening rules**, not official
Birmingham City Council classifications.

### Air-pollution screening

The air component is deliberately kept separate from the measured
DEFRA station trend.

Birmingham's road data identify classified A and B roads.

An area within the 100 m project screening corridor of an A Road
is flagged as **HIGH road-air exposure screening**.

An area within the 100 m corridor of a B Road is flagged as
**MODERATE road-air exposure screening**.

Areas without these road intersections are flagged **LOW**.

This does not claim that every point within 100 m has a particular
NO₂ concentration. It identifies locations where road exposure
provides a reasonable reason for further air-quality investigation.

The measured DEFRA Birmingham Ladywood NO₂ and PM₂.₅ data are shown
separately as the actual monitoring evidence.
"""
)


# ============================================================
# DOWNLOADABLE LADYWOOD DATASET
# ============================================================

st.subheader(
    "Ladywood environmental results"
)

download_columns = [
    column
    for column in [
        "LSOA21CD",
        "LSOA21NM",
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
    if column in results.columns
]

download_data = results[
    download_columns
].copy()

csv_data = download_data.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Ladywood environmental results CSV",
    data=csv_data,
    file_name="Ladywood_Environmental_Results.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Ladywood study boundary: Birmingham City Council ward data. "
    "Spatial evidence: Birmingham City Council CRVA 2025, highways "
    "and Brownfield Register 2025. Air monitoring: DEFRA UK-AIR "
    "Birmingham Ladywood monitoring site BMLD."
)
