import streamlit as st
import pandas as pd
import numpy as np
import requests
import geopandas as gpd
import folium
import plotly.express as px

from io import StringIO
from shapely.geometry import shape
from shapely.ops import unary_union
from streamlit_folium import st_folium


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Risk Dashboard",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# OFFICIAL DATA SOURCES
# ============================================================

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

WARD_LAYER = f"{CRVA}/14"
CRVA_LSOA_LAYER = f"{CRVA}/15"

FLOOD_ZONE_3 = f"{CRVA}/109"
SURFACE_FLOOD = f"{CRVA}/1546"

BROWNFIELD = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_LandUse/MapServer/38"
)

# UK Government DEFRA UK-AIR.
# Birmingham Ladywood monitoring site = BMLD
AIR_BASE = (
    "https://uk-air.defra.gov.uk/datastore/data_files/"
    "site_pol_data/"
)

REQUEST_TIMEOUT = 90


# ============================================================
# BASIC HELPERS
# ============================================================

def fix_geometry(gdf):
    """
    Repair invalid geometries before any spatial operation.

    This prevents the TopologyException / side location
    conflict error produced by invalid polygons.
    """

    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    if "geometry" not in gdf.columns:
        return gdf

    gdf = gdf[gdf.geometry.notna()].copy()

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
    fields="*"
):

    params = {
        "where": where,
        "outFields": fields,
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson"
    }

    response = requests.get(
        layer_url + "/query",
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    features = data.get("features", [])

    rows = []

    for feature in features:

        properties = feature.get(
            "properties",
            {}
        )

        geom = feature.get(
            "geometry"
        )

        if geom:

            try:
                geometry = shape(geom)
            except Exception:
                geometry = None

        else:
            geometry = None

        row = properties.copy()

        row["geometry"] = geometry

        rows.append(row)

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

    return fix_geometry(gdf)


# ============================================================
# LADYWOOD
# ============================================================

@st.cache_data(ttl=3600)
def get_ladywood():

    data = arcgis_query(
        WARD_LAYER,
        where="UPPER(WARDNME) = 'LADYWOOD'",
        fields="WARDNME,WARD_CODE,MEAN,MIN,MAX"
    )

    if data.empty:

        data = arcgis_query(
            WARD_LAYER,
            where="1=1",
            fields="WARDNME,WARD_CODE,MEAN,MIN,MAX"
        )

        data = data[
            data["WARDNME"]
            .astype(str)
            .str.contains(
                "Ladywood",
                case=False,
                na=False
            )
        ].copy()

    if data.empty:
        raise RuntimeError(
            "The official Birmingham ward service "
            "did not return Ladywood."
        )

    return fix_geometry(data)


# ============================================================
# CRVA
# ============================================================

@st.cache_data(ttl=3600)
def get_crva_lsoa():

    data = arcgis_query(
        CRVA_LSOA_LAYER,
        where="1=1",
        fields=(
            "LSOA21CD,LSOA21NM,MIN,MAX,MEAN,"
            "STD,MEDIAN,MINIMUM_RISK,"
            "AVERAGE_RISK,MAXIMUM_RISK"
        )
    )

    if data.empty:
        raise RuntimeError(
            "The Birmingham CRVA LSOA service returned no data."
        )

    return fix_geometry(data)


# ============================================================
# FLOODING
# ============================================================

@st.cache_data(ttl=3600)
def get_flood_zone_3():

    return arcgis_query(
        FLOOD_ZONE_3,
        fields="origin,flood_zone,flood_sour"
    )


@st.cache_data(ttl=3600)
def get_surface_flood():

    return arcgis_query(
        SURFACE_FLOOD,
        fields="pub_date,tile_id"
    )


# ============================================================
# BROWNFIELD
# ============================================================

@st.cache_data(ttl=3600)
def get_brownfield():

    return arcgis_query(
        BROWNFIELD,
        fields=(
            "SiteReference,SiteNameAddress,"
            "GeoX,GeoY,Hectares,"
            "OwnershipStatus,PlanningStatus,"
            "PermissionType,PlanningHistory,"
            "HazardousSubstances,FirstAddedDate,"
            "LastUpdatedDate"
        )
    )


# ============================================================
# SPATIAL PREPARATION
# ============================================================

def prepare_ladywood_indicators(
    ladywood,
    lsoa,
    flood3,
    surface,
    brownfield
):

    # British National Grid
    ladywood = fix_geometry(
        ladywood.to_crs(27700)
    )

    lsoa = fix_geometry(
        lsoa.to_crs(27700)
    )

    flood3 = fix_geometry(
        flood3.to_crs(27700)
    )

    surface = fix_geometry(
        surface.to_crs(27700)
    )

    brownfield = fix_geometry(
        brownfield.to_crs(27700)
    )

    ladywood_geom = unary_union(
        ladywood.geometry
    )

    # --------------------------------------------------------
    # KEEP ONLY LSOAs THAT INTERSECT LADYWOOD
    # --------------------------------------------------------

    lsoa = lsoa[
        lsoa.geometry.intersects(
            ladywood_geom
        )
    ].copy()

    if lsoa.empty:
        raise RuntimeError(
            "No CRVA LSOAs intersect Ladywood."
        )

    # --------------------------------------------------------
    # CLIP TO LADYWOOD
    # --------------------------------------------------------

    clipped_geometries = []

    for geom in lsoa.geometry:

        try:

            clipped = geom.intersection(
                ladywood_geom
            )

        except Exception:

            clipped = geom.make_valid().intersection(
                ladywood_geom
            )

        clipped_geometries.append(
            clipped
        )

    lsoa["geometry"] = clipped_geometries

    lsoa = lsoa[
        lsoa.geometry.notna()
        & ~lsoa.geometry.is_empty
    ].copy()

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    lsoa["Area_m2"] = (
        lsoa.geometry.area
    )

    # --------------------------------------------------------
    # FLOOD ZONE 3
    # --------------------------------------------------------

    if not flood3.empty:

        flood_union = unary_union(
            flood3.geometry
        )

        flood_areas = []

        for geom in lsoa.geometry:

            try:
                area = geom.intersection(
                    flood_union
                ).area

            except Exception:

                area = geom.make_valid().intersection(
                    flood_union
                ).area

            flood_areas.append(area)

        lsoa["Flood3_m2"] = flood_areas

    else:

        lsoa["Flood3_m2"] = 0.0

    lsoa["Flood3_pct"] = np.where(
        lsoa["Area_m2"] > 0,
        lsoa["Flood3_m2"]
        / lsoa["Area_m2"] * 100,
        0
    )

    # --------------------------------------------------------
    # SURFACE WATER
    # --------------------------------------------------------

    if not surface.empty:

        surface_union = unary_union(
            surface.geometry
        )

        surface_areas = []

        for geom in lsoa.geometry:

            try:

                area = geom.intersection(
                    surface_union
                ).area

            except Exception:

                area = geom.make_valid().intersection(
                    surface_union
                ).area

            surface_areas.append(area)

        lsoa["SurfaceFlood_m2"] = surface_areas

    else:

        lsoa["SurfaceFlood_m2"] = 0.0

    lsoa["SurfaceFlood_pct"] = np.where(
        lsoa["Area_m2"] > 0,
        lsoa["SurfaceFlood_m2"]
        / lsoa["Area_m2"] * 100,
        0
    )

    # --------------------------------------------------------
    # COMBINED FLOOD EXPOSURE
    #
    # Do not double count overlapping flood layers.
    # --------------------------------------------------------

    lsoa["FloodExposure_pct"] = lsoa[
        [
            "Flood3_pct",
            "SurfaceFlood_pct"
        ]
    ].max(axis=1)

    # --------------------------------------------------------
    # BROWNFIELD
    # --------------------------------------------------------

    if not brownfield.empty:

        brownfield_union = unary_union(
            brownfield.geometry
        )

        brownfield_area = []

        brownfield_count = []

        for geom in lsoa.geometry:

            try:

                area = geom.intersection(
                    brownfield_union
                ).area

            except Exception:

                area = geom.make_valid().intersection(
                    brownfield_union
                ).area

            count = int(
                brownfield.geometry.intersects(
                    geom
                ).sum()
            )

            brownfield_area.append(
                area
            )

            brownfield_count.append(
                count
            )

        lsoa["Brownfield_m2"] = (
            brownfield_area
        )

        lsoa["Brownfield_sites"] = (
            brownfield_count
        )

    else:

        lsoa["Brownfield_m2"] = 0.0

        lsoa["Brownfield_sites"] = 0

    lsoa["Brownfield_pct"] = np.where(
        lsoa["Area_m2"] > 0,
        lsoa["Brownfield_m2"]
        / lsoa["Area_m2"] * 100,
        0
    )

    # --------------------------------------------------------
    # OFFICIAL CRVA
    # --------------------------------------------------------

    lsoa["CRVA"] = pd.to_numeric(
        lsoa["MEAN"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # NORMALISE ONLY WITHIN LADYWOOD
    #
    # No arbitrary external multiplier.
    # --------------------------------------------------------

    def normalise(series):

        series = pd.to_numeric(
            series,
            errors="coerce"
        ).fillna(0)

        low = series.min()
        high = series.max()

        if high == low:
            return pd.Series(
                0.0,
                index=series.index
            )

        return (
            (series - low)
            / (high - low)
        )

    lsoa["CRVA_index"] = normalise(
        lsoa["CRVA"]
    )

    lsoa["Flood_index"] = normalise(
        lsoa["FloodExposure_pct"]
    )

    lsoa["Brownfield_index"] = normalise(
        lsoa["Brownfield_pct"]
    )

    # --------------------------------------------------------
    # TRANSPARENT ENVIRONMENTAL SCREENING
    #
    # Equal contribution from three spatial evidence groups.
    #
    # This is OUR PROJECT SCREENING INDEX.
    # It is NOT a Birmingham Council classification.
    # --------------------------------------------------------

    lsoa["Environmental_Index"] = (
        lsoa["CRVA_index"]
        + lsoa["Flood_index"]
        + lsoa["Brownfield_index"]
    ) / 3

    lsoa["Environmental_Percent"] = (
        lsoa["Environmental_Index"] * 100
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    def risk_class(value):

        if value >= 0.67:
            return "HIGH"

        if value >= 0.34:
            return "MODERATE"

        return "LOW"

    lsoa["Risk"] = (
        lsoa["Environmental_Index"]
        .apply(risk_class)
    )

    # --------------------------------------------------------
    # CENTROIDS
    # --------------------------------------------------------

    centroids = (
        lsoa.geometry.centroid
    )

    lsoa["Longitude"] = (
        centroids.x
    )

    lsoa["Latitude"] = (
        centroids.y
    )

    return lsoa


# ============================================================
# DEFRA LADYWOOD AIR QUALITY
# ============================================================

def read_defra_file(url):

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    text = response.text

    # DEFRA CSV files can have metadata/header
    # before the actual data. Try several approaches.

    try:

        df = pd.read_csv(
            StringIO(text)
        )

        if len(df.columns) > 1:
            return df

    except Exception:
        pass

    lines = text.splitlines()

    data_start = None

    for i, line in enumerate(lines):

        lower = line.lower()

        if (
            "date" in lower
            and "time" in lower
        ):

            data_start = i

            break

    if data_start is None:
        raise RuntimeError(
            "Could not identify the DEFRA CSV data header."
        )

    return pd.read_csv(
        StringIO(
            "\n".join(
                lines[data_start:]
            )
        )
    )


def clean_air_dataframe(
    df,
    pollutant
):

    df = df.copy()

    # Find date column
    date_col = None

    for col in df.columns:

        name = str(col).lower()

        if (
            "date" in name
            or "datetime" in name
        ):

            date_col = col

            break

    if date_col is None:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # Find concentration column
    candidate = None

    preferred = [
        "Value",
        "value",
        pollutant,
        pollutant.upper(),
        "Concentration",
        "concentration"
    ]

    for name in preferred:

        if name in df.columns:

            candidate = name

            break

    if candidate is None:

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns
        )

        if len(numeric_columns) == 0:
            return pd.DataFrame()

        candidate = numeric_columns[-1]

    df["Concentration"] = pd.to_numeric(
        df[candidate],
        errors="coerce"
    )

    df = df[
        [
            "Date",
            "Concentration"
        ]
    ].dropna()

    df["Pollutant"] = pollutant

    return df


@st.cache_data(ttl=3600)
def get_air_quality():

    all_data = []

    years = [
        2019,
        2020,
        2021,
        2022,
        2023,
        2024,
        2025,
        2026
    ]

    for year in years:

        urls = {
            "NO2": (
                f"{AIR_BASE}"
                f"BMLD_NO2_{year}.csv"
            ),

            "PM2.5": (
                f"{AIR_BASE}"
                f"BMLD_PM25_{year}.csv"
            )
        }

        for pollutant, url in urls.items():

            try:

                raw = read_defra_file(
                    url
                )

                cleaned = clean_air_dataframe(
                    raw,
                    pollutant
                )

                if not cleaned.empty:

                    all_data.append(
                        cleaned
                    )

            except Exception:

                # A missing/incomplete year is not
                # replaced with zero.
                continue

    if not all_data:

        return pd.DataFrame(
            columns=[
                "Date",
                "Concentration",
                "Pollutant"
            ]
        )

    air = pd.concat(
        all_data,
        ignore_index=True
    )

    air["Year"] = (
        air["Date"].dt.year
    )

    air["Month"] = (
        air["Date"].dt.to_period(
            "M"
        ).astype(str)
    )

    return air


# ============================================================
# LOAD
# ============================================================

st.title(
    "LADYWOOD"
)

st.subheader(
    "Environmental Risk Dashboard"
)

st.caption(
    "Ladywood, Birmingham, United Kingdom"
)

st.info(
    "This dashboard uses live data from Birmingham City "
    "Council and the UK Government DEFRA UK-AIR Ladywood "
    "monitoring site. It does not use the supplied Excel file."
)


try:

    with st.spinner(
        "Loading official Ladywood environmental data..."
    ):

        ladywood = get_ladywood()

        crva = get_crva_lsoa()

        flood3 = get_flood_zone_3()

        surface = get_surface_flood()

        brownfield = get_brownfield()

        indicators = prepare_ladywood_indicators(
            ladywood,
            crva,
            flood3,
            surface,
            brownfield
        )

        air = get_air_quality()


except Exception as error:

    st.error(
        "The dashboard could not load the official data."
    )

    st.exception(error)

    st.stop()


# ============================================================
# TOP SUMMARY
# ============================================================

st.header(
    "Ladywood Environmental Overview"
)

high_count = int(
    (indicators["Risk"] == "HIGH").sum()
)

moderate_count = int(
    (indicators["Risk"] == "MODERATE").sum()
)

brownfield_sites = int(
    indicators["Brownfield_sites"].sum()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Areas assessed",
    len(indicators)
)

c2.metric(
    "High-priority areas",
    high_count
)

c3.metric(
    "Moderate areas",
    moderate_count
)

c4.metric(
    "Brownfield sites intersecting areas",
    brownfield_sites
)


# ============================================================
# MAP
# ============================================================

st.header(
    "Ladywood Environmental Risk Map"
)

st.write(
    "Green, yellow and red markers identify the relative "
    "environmental screening level of official 2021 LSOA "
    "areas intersecting the Ladywood ward."
)

ladywood_map = ladywood.to_crs(
    4326
)

ladywood_shape = unary_union(
    ladywood_map.geometry
)

centre = ladywood_shape.centroid

m = folium.Map(
    location=[
        centre.y,
        centre.x
    ],
    zoom_start=13,
    tiles="OpenStreetMap"
)


# ------------------------------------------------------------
# LADYWOOD BOUNDARY
# ------------------------------------------------------------

folium.GeoJson(
    ladywood_map.to_json(),
    name="Ladywood boundary",
    style_function=lambda feature: {
        "color": "black",
        "weight": 4,
        "fillOpacity": 0
    }
).add_to(m)


# ------------------------------------------------------------
# FLOOD ZONE
# ------------------------------------------------------------

if not flood3.empty:

    flood_map = flood3.to_crs(
        4326
    )

    folium.GeoJson(
        flood_map.to_json(),
        name="Flood Zone 3",
        style_function=lambda feature: {
            "color": "blue",
            "fillColor": "blue",
            "weight": 1,
            "fillOpacity": 0.25
        }
    ).add_to(m)


# ------------------------------------------------------------
# SURFACE WATER
# ------------------------------------------------------------

if not surface.empty:

    surface_map = surface.to_crs(
        4326
    )

    folium.GeoJson(
        surface_map.to_json(),
        name="Surface-water flood risk",
        style_function=lambda feature: {
            "color": "cyan",
            "fillColor": "cyan",
            "weight": 1,
            "fillOpacity": 0.18
        }
    ).add_to(m)


# ------------------------------------------------------------
# BROWNFIELD
# ------------------------------------------------------------

if not brownfield.empty:

    brownfield_map = brownfield.to_crs(
        4326
    )

    folium.GeoJson(
        brownfield_map.to_json(),
        name="Brownfield Register 2025",
        style_function=lambda feature: {
            "color": "brown",
            "fillColor": "brown",
            "weight": 2,
            "fillOpacity": 0.35
        }
    ).add_to(m)


# ------------------------------------------------------------
# RISK DOTS
# ------------------------------------------------------------

for _, row in indicators.iterrows():

    if row["Risk"] == "HIGH":

        colour = "red"

    elif row["Risk"] == "MODERATE":

        colour = "orange"

    else:

        colour = "green"

    popup = f"""
    <div style="width:300px">

    <h4>{row['LSOA21NM']}</h4>

    <b>Environmental screening:</b>
    {row['Risk']}<br><br>

    <b>Official CRVA mean:</b>
    {row['CRVA']:.2f}<br>

    <b>CRVA risk:</b>
    {row['AVERAGE_RISK']}<br><br>

    <b>Flood Zone 3:</b>
    {row['Flood3_pct']:.2f}%<br>

    <b>Surface-water exposure:</b>
    {row['SurfaceFlood_pct']:.2f}%<br>

    <b>Combined flood exposure:</b>
    {row['FloodExposure_pct']:.2f}%<br><br>

    <b>Brownfield coverage:</b>
    {row['Brownfield_pct']:.2f}%<br>

    <b>Brownfield sites:</b>
    {int(row['Brownfield_sites'])}<br><br>

    <b>Project screening score:</b>
    {row['Environmental_Percent']:.1f}%

    </div>
    """

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=9,
        color=colour,
        fill=True,
        fill_color=colour,
        fill_opacity=0.9,
        popup=folium.Popup(
            popup,
            max_width=350
        ),
        tooltip=(
            f"{row['LSOA21NM']} — "
            f"{row['Risk']}"
        )
    ).add_to(m)


folium.LayerControl().add_to(m)

st_folium(
    m,
    width=None,
    height=650
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.header(
    "Ladywood Risk Distribution"
)

risk_counts = (
    indicators["Risk"]
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

risk_counts.columns = [
    "Risk level",
    "Number of areas"
]

fig_risk = px.bar(
    risk_counts,
    x="Risk level",
    y="Number of areas",
    title="Environmental screening classification"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# CRVA
# ============================================================

st.header(
    "Climate Risk and Vulnerability Assessment"
)

st.write(
    "The CRVA is Birmingham City Council's official "
    "climate-risk assessment. Birmingham states that it "
    "combines multiple climate-related factors including "
    "NO₂, PM2.5, fluvial flooding, surface flooding, "
    "temperature, green-space deficit, tree-canopy deficit "
    "and vulnerability indicators."
)

crva_chart = indicators.sort_values(
    "CRVA"
)

fig_crva = px.bar(
    crva_chart,
    x="CRVA",
    y="LSOA21NM",
    orientation="h",
    title="Official Birmingham CRVA mean",
    labels={
        "CRVA": "CRVA mean",
        "LSOA21NM": "Ladywood statistical area"
    }
)

st.plotly_chart(
    fig_crva,
    use_container_width=True
)


# ============================================================
# FLOODING
# ============================================================

st.header(
    "Flood Risk"
)

flood_long = indicators[
    [
        "LSOA21NM",
        "Flood3_pct",
        "SurfaceFlood_pct"
    ]
].melt(
    id_vars="LSOA21NM",
    var_name="Flood indicator",
    value_name="Area percentage"
)

flood_long[
    "Flood indicator"
] = flood_long[
    "Flood indicator"
].replace(
    {
        "Flood3_pct": "Flood Zone 3",
        "SurfaceFlood_pct": "Surface water"
    }
)

fig_flood = px.bar(
    flood_long,
    x="LSOA21NM",
    y="Area percentage",
    color="Flood indicator",
    barmode="group",
    title="Flood exposure within Ladywood"
)

st.plotly_chart(
    fig_flood,
    use_container_width=True
)


# ============================================================
# BROWNFIELD
# ============================================================

st.header(
    "Brownfield / Previously Developed Land"
)

fig_brown = px.bar(
    indicators.sort_values(
        "Brownfield_pct",
        ascending=False
    ),
    x="LSOA21NM",
    y="Brownfield_pct",
    title="Brownfield coverage by Ladywood statistical area",
    labels={
        "LSOA21NM": "Ladywood area",
        "Brownfield_pct": "Brownfield coverage (%)"
    }
)

st.plotly_chart(
    fig_brown,
    use_container_width=True
)


# ============================================================
# AIR QUALITY
# ============================================================

st.header(
    "Ladywood Air Quality"
)

if air.empty:

    st.warning(
        "DEFRA UK-AIR did not return readable Ladywood "
        "monitoring data at this time."
    )

else:

    air_monthly = (
        air.groupby(
            [
                "Month",
                "Pollutant"
            ],
            as_index=False
        )["Concentration"]
        .mean()
    )

    fig_air = px.line(
        air_monthly,
        x="Month",
        y="Concentration",
        color="Pollutant",
        markers=True,
        title=(
            "Monthly mean pollutant concentration — "
            "DEFRA Birmingham Ladywood monitoring site"
        ),
        labels={
            "Month": "Month",
            "Concentration": "Concentration",
            "Pollutant": "Pollutant"
        }
    )

    st.plotly_chart(
        fig_air,
        use_container_width=True
    )

    # --------------------------------------------------------
    # YEARLY
    # --------------------------------------------------------

    air_yearly = (
        air.groupby(
            [
                "Year",
                "Pollutant"
            ],
            as_index=False
        )["Concentration"]
        .mean()
    )

    fig_air_year = px.line(
        air_yearly,
        x="Year",
        y="Concentration",
        color="Pollutant",
        markers=True,
        title=(
            "Annual mean pollutant concentration — "
            "Birmingham Ladywood monitoring site"
        )
    )

    st.plotly_chart(
        fig_air_year,
        use_container_width=True
    )

    st.caption(
        "Air measurements are from the Birmingham Ladywood "
        "DEFRA monitoring station. They are not artificially "
        "redistributed to individual neighbourhoods."
    )


# ============================================================
# COMBINED SCREENING
# ============================================================

st.header(
    "Combined Environmental Screening"
)

combined = indicators[
    [
        "LSOA21NM",
        "CRVA",
        "AVERAGE_RISK",
        "FloodExposure_pct",
        "Brownfield_pct",
        "Brownfield_sites",
        "Environmental_Percent",
        "Risk"
    ]
].copy()

combined.columns = [
    "Ladywood area",
    "CRVA mean",
    "Official CRVA classification",
    "Flood exposure (%)",
    "Brownfield coverage (%)",
    "Brownfield sites",
    "Environmental screening (%)",
    "Risk"
]

combined = combined.sort_values(
    "Environmental screening (%)",
    ascending=False
)

st.dataframe(
    combined,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# COMBINED GRAPH
# ============================================================

fig_combined = px.bar(
    combined.sort_values(
        "Environmental screening (%)"
    ),
    x="Environmental screening (%)",
    y="Ladywood area",
    color="Risk",
    orientation="h",
    title=(
        "Ladywood combined environmental "
        "screening indicator"
    )
)

st.plotly_chart(
    fig_combined,
    use_container_width=True
)


# ============================================================
# HIGHEST PRIORITY
# ============================================================

st.header(
    "Priority Areas"
)

top = combined.head(
    min(5, len(combined))
)

for _, row in top.iterrows():

    st.write(
        f"**{row['Ladywood area']}** — "
        f"{row['Risk']} priority "
        f"({row['Environmental screening (%)']:.1f}%)"
    )


# ============================================================
# METHODOLOGY
# ============================================================

st.header(
    "How the screening calculation works"
)

st.write(
    """
    The dashboard does not invent pollution values or assign
    arbitrary zone multipliers.

    Three spatial evidence groups are used:

    1. Birmingham's official Climate Risk and Vulnerability
       Assessment (CRVA).

    2. Official mapped flood exposure from Birmingham's
       Flood Zone 3 and surface-water datasets.

    3. Official Birmingham Brownfield Register 2025.

    Each spatial indicator is normalised using the range
    actually observed within the Ladywood study area.

    The three normalised indicators are then given equal
    contribution to produce an Environmental Screening Index.

    This index is a project screening calculation. It is NOT
    an official Birmingham City Council risk classification.

    The red, orange and green markers therefore identify areas
    with stronger, intermediate or weaker combined evidence
    within the Ladywood study area.
    """
)


# ============================================================
# DATA TABLE
# ============================================================

st.header(
    "Ladywood Environmental Dataset"
)

st.dataframe(
    indicators[
        [
            "LSOA21NM",
            "CRVA",
            "AVERAGE_RISK",
            "Flood3_pct",
            "SurfaceFlood_pct",
            "FloodExposure_pct",
            "Brownfield_pct",
            "Brownfield_sites",
            "Environmental_Percent",
            "Risk"
        ]
    ].sort_values(
        "Environmental_Percent",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD
# ============================================================

download = combined.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    "Download Ladywood environmental results",
    download,
    "Ladywood_environmental_results.csv",
    "text/csv"
)


# ============================================================
# SOURCES
# ============================================================

st.header(
    "Primary Sources"
)

st.markdown(
    """
    **Birmingham City Council**

    • Climate Risk and Vulnerability Assessment (CRVA) 2025

    • Flood Risk Zone 3

    • Surface-water flood risk

    • Brownfield Register 2025

    • Birmingham ward boundaries


    **UK Government / DEFRA UK-AIR**

    • Birmingham Ladywood automatic monitoring station

    • NO₂ monitoring

    • PM2.5 monitoring


    No supplied Excel dataset is required by this dashboard.
    """
)

st.caption(
    "Ladywood is the spatial study area. Statistical areas "
    "shown on the map are official 2021 LSOAs intersecting "
    "the Ladywood ward boundary."
)
