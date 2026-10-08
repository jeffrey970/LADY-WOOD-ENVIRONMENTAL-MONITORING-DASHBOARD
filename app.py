import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# LADYWOOD AIR-RISK DASHBOARD
# Data-driven Streamlit application
# ============================================================

st.set_page_config(
    page_title="Ladywood Air Risk Dashboard",
    page_icon="ðŸŒ±",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """ <style> .main {background-color: #f7f9fb;} .block-container {padding-top: 1.5rem;} .risk-high { padding: 12px; border-radius: 8px; background-color: #ffe5e5; border-left: 5px solid #d62728; } .risk-moderate { padding: 12px; border-radius: 8px; background-color: #fff4d6; border-left: 5px solid #e6a700; } .risk-low { padding: 12px; border-radius: 8px; background-color: #e5f6e9; border-left: 5px solid #2ca02c; } </style> """,
    unsafe_allow_html=True,
)

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_FILE = "Ladywood_4_Zone_Air_Risk_Final.xlsx"

ZONE_MULTIPLIERS = {
    "City Centre": 1.15,
    "Jewellery Quarter & Broad Street": 1.20,
    "Park Central": 1.00,
    "Remainder of Ladywood": 0.95,
}

WHO_GUIDELINES = {
    "NO2": 10.0,
    "PM2.5": 5.0,
    "PM10": 15.0,
}

RISK_THRESHOLDS = {
    "LOW_MAX": 100.0,
    "MODERATE_MAX": 150.0,
}

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_workbook(file_path: str):
    """Load the workbook sheets used by the dashboard."""
    xls = pd.ExcelFile(file_path)

    required = {
        "Original_Monthly_Data",
        "Station_Annual_Calcs",
        "Zone_Year_Calcs",
        "Zone_Risk_Summary",
        "Methodology",
    }

    missing = required.difference(xls.sheet_names)
    if missing:
        raise ValueError(
            f"Workbook is missing required sheets: {', '.join(sorted(missing))}"
        )

    return {
        "monthly": pd.read_excel(file_path, sheet_name="Original_Monthly_Data"),
        "station": pd.read_excel(file_path, sheet_name="Station_Annual_Calcs"),
        "zone_year": pd.read_excel(file_path, sheet_name="Zone_Year_Calcs"),
        "zone_summary": pd.read_excel(file_path, sheet_name="Zone_Risk_Summary"),
        "methodology": pd.read_excel(file_path, sheet_name="Methodology"),
    }


def weighted_mean(values, valid_counts):
    """Calculate an observation-weighted mean, ignoring unavailable data."""
    values = pd.to_numeric(values, errors="coerce")
    valid_counts = pd.to_numeric(valid_counts, errors="coerce")

    mask = values.notna() & valid_counts.notna() & (valid_counts > 0)

    if not mask.any():
        return np.nan

    return np.average(values[mask], weights=valid_counts[mask])


def calculate_station_annual(monthly):
    """Recalculate annual station means from monthly data."""
    rows = []

    for year, group in monthly.groupby("Year", sort=True):
        rows.append(
            {
                "Year": int(year),
                "NO2": weighted_mean(
                    group["NO2_Average"], group["NO2_Valid_Observations"]
                ),
                "NO2_Valid_Observations": group["NO2_Valid_Observations"].sum(),
                "PM2.5": weighted_mean(
                    group["PM2_5_Average"], group["PM2_5_Valid_Observations"]
                ),
                "PM2.5_Valid_Observations": group[
                    "PM2_5_Valid_Observations"
                ].sum(),
                "PM10": weighted_mean(
                    group["PM10_Average"], group["PM10_Valid_Observations"]
                ),
                "PM10_Valid_Observations": group[
                    "PM10_Valid_Observations"
                ].sum(),
            }
        )

    return pd.DataFrame(rows)


def risk_class(no2_percent, pm25_percent):
    """Dashboard screening classification."""
    if pd.isna(no2_percent) or pd.isna(pm25_percent):
        return "NO DATA"

    score = (no2_percent + pm25_percent) / 2

    if score <= RISK_THRESHOLDS["LOW_MAX"]:
        return "LOW"
    elif score <= RISK_THRESHOLDS["MODERATE_MAX"]:
        return "MODERATE"
    return "HIGH"


def calculate_zone_results(station_annual):
    """Calculate estimated zone concentrations and risk from station data."""
    rows = []

    for _, station in station_annual.iterrows():
        year = int(station["Year"])

        for zone, multiplier in ZONE_MULTIPLIERS.items():
            no2 = station["NO2"] * multiplier
            pm25 = station["PM2.5"] * multiplier
            pm10 = station["PM10"] * multiplier

            no2_pct = (no2 / WHO_GUIDELINES["NO2"]) * 100
            pm25_pct = (pm25 / WHO_GUIDELINES["PM2.5"]) * 100
            pm10_pct = (pm10 / WHO_GUIDELINES["PM10"]) * 100

            score = (no2_pct + pm25_pct) / 2

            rows.append(
                {
                    "Year": year,
                    "Zone": zone,
                    "Air_Multiplier": multiplier,
                    "NO2_Station": station["NO2"],
                    "NO2_Calculated": no2,
                    "NO2_WHO_Percent": no2_pct,
                    "PM2.5_Station": station["PM2.5"],
                    "PM2.5_Calculated": pm25,
                    "PM2.5_WHO_Percent": pm25_pct,
                    "PM10_Station": station["PM10"],
                    "PM10_Calculated": pm10,
                    "PM10_WHO_Percent": pm10_pct,
                    "Air_Risk_Score": score,
                    "Air_Risk_Class": risk_class(no2_pct, pm25_pct),
                }
            )

    return pd.DataFrame(rows)


def risk_colour_class(value):
    if value == "HIGH":
        return "risk-high"
    if value == "MODERATE":
        return "risk-moderate"
    if value == "LOW":
        return "risk-low"
    return ""


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("ðŸŒ± Ladywood Air Risk")
st.sidebar.caption("Data-driven environmental screening dashboard")

uploaded = st.sidebar.file_uploader(
    "Use another Ladywood Excel workbook",
    type=["xlsx"],
    help="Upload a workbook with the same required sheet structure.",
)

if uploaded is not None:
    workbook_source = uploaded
else:
    workbook_source = DEFAULT_FILE

if isinstance(workbook_source, str) and not Path(workbook_source).exists():
    st.error(
        f"Could not find `{DEFAULT_FILE}`. Put the Excel file in the same folder as app.py."
    )
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
try:
    data = load_workbook(workbook_source)
except Exception as exc:
    st.error(f"Could not load the workbook: {exc}")
    st.stop()

monthly = data["monthly"].copy()
station_original = data["station"].copy()
zone_original = data["zone_year"].copy()
summary_original = data["zone_summary"].copy()
methodology = data["methodology"].copy()

# Recalculate rather than trusting hard-coded dashboard outputs
station_calc = calculate_station_annual(monthly)
zone_calc = calculate_zone_results(station_calc)

# Ensure numeric fields are numeric
for frame in [monthly, station_calc, zone_calc, summary_original]:
    for column in frame.columns:
        if column not in ["Month", "Year_Month", "Zone", "Air_Risk_Class"]:
            frame[column] = pd.to_numeric(frame[column], errors="ignore")

# ============================================================
# HEADER
# ============================================================
st.title("Ladywood Interactive Air-Risk Dashboard")
st.markdown(
    """ **Purpose:** use Ladywood monitoring data to identify air-pollution trends, compare the four development-plan zones, and transparently screen relative air-risk areas. """
)

st.info(
    "Important: the monitoring station is a spatial measurement point. "
    "Zone concentrations shown here are calculated estimates using the project "
    "screening multipliers; they are not direct measurements inside each zone."
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================
available_years = sorted(station_calc["Year"].dropna().astype(int).unique())
selected_year = st.sidebar.selectbox(
    "Analysis year",
    available_years,
    index=len(available_years) - 1,
)

selected_zone = st.sidebar.selectbox(
    "Zone",
    ["All zones"] + list(ZONE_MULTIPLIERS.keys()),
)

# ============================================================
# SELECTED DATA
# ============================================================
year_station = station_calc[station_calc["Year"] == selected_year].iloc[0]
year_zones = zone_calc[zone_calc["Year"] == selected_year].copy()

if selected_zone != "All zones":
    displayed_zones = year_zones[year_zones["Zone"] == selected_zone].copy()
else:
    displayed_zones = year_zones.copy()

# ============================================================
# KPI ROW
# ============================================================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Station NOâ‚‚", f"{year_station['NO2']:.2f} Âµg/mÂ³")

with c2:
    st.metric("Station PMâ‚‚.â‚…", f"{year_station['PM2.5']:.2f} Âµg/mÂ³")

with c3:
    st.metric("Station PMâ‚â‚€", f"{year_station['PM10']:.2f} Âµg/mÂ³")

with c4:
    highest = year_zones.sort_values("Air_Risk_Score", ascending=False).iloc[0]
    st.metric(
        "Highest estimated zone risk",
        highest["Zone"],
        f"{highest['Air_Risk_Score']:.1f} score",
    )

# ============================================================
# MAIN TABS
# ============================================================
tab_overview, tab_trends, tab_zones, tab_data, tab_method = st.tabs(
    [
        "ðŸ“Š Overview",
        "ðŸ“ˆ Trends",
        "ðŸ“ Zone Risk",
        "ðŸ”Ž Data & Calculations",
        "ðŸ“š Methodology",
    ]
)

# ============================================================
# OVERVIEW
# ============================================================
with tab_overview:
    st.subheader(f"Air-risk overview â€” {selected_year}")

    col_a, col_b = st.columns(2)

    with col_a:
        chart = px.bar(
            displayed_zones,
            x="Zone",
            y="Air_Risk_Score",
            color="Air_Risk_Class",
            text="Air_Risk_Score",
            title="Estimated air-risk score by zone",
            labels={"Air_Risk_Score": "Risk score (%)"},
        )
        chart.add_hline(
            y=100,
            line_dash="dash",
            annotation_text="100% threshold",
        )
        chart.add_hline(
            y=150,
            line_dash="dash",
            annotation_text="150% threshold",
        )
        chart.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        st.plotly_chart(chart, use_container_width=True)

    with col_b:
        pollutant_data = displayed_zones[
            ["Zone", "NO2_WHO_Percent", "PM2.5_WHO_Percent", "PM10_WHO_Percent"]
        ].melt(
            id_vars="Zone",
            var_name="Pollutant",
            value_name="WHO_Percent",
        )

        pollutant_data["Pollutant"] = pollutant_data["Pollutant"].str.replace(
            "_WHO_Percent", "", regex=False
        )

        chart2 = px.bar(
            pollutant_data,
            x="Zone",
            y="WHO_Percent",
            color="Pollutant",
            barmode="group",
            title="Estimated pollutant concentration relative to WHO guideline",
            labels={"WHO_Percent": "Percentage of WHO guideline"},
        )
        chart2.add_hline(y=100, line_dash="dash", annotation_text="100%")
        st.plotly_chart(chart2, use_container_width=True)

    st.subheader("Priority zones")

    ranking = year_zones.sort_values("Air_Risk_Score", ascending=False).copy()
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))

    st.dataframe(
        ranking[
            [
                "Rank",
                "Zone",
                "Air_Multiplier",
                "NO2_Calculated",
                "PM2.5_Calculated",
                "Air_Risk_Score",
                "Air_Risk_Class",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# TRENDS
# ============================================================
with tab_trends:
    st.subheader("Ladywood station pollutant trends")

    pollutant = st.selectbox(
        "Pollutant",
        ["NO2", "PM2.5", "PM10"],
        key="trend_pollutant",
    )

    trend = station_calc[["Year", pollutant]].copy()

    fig = px.line(
        trend,
        x="Year",
        y=pollutant,
        markers=True,
        title=f"Annual {pollutant} trend at the Ladywood monitoring station",
        labels={pollutant: f"{pollutant} (Âµg/mÂ³)"},
    )

    guideline = WHO_GUIDELINES[pollutant]
    fig.add_hline(
        y=guideline,
        line_dash="dash",
        annotation_text=f"WHO annual guideline: {guideline} Âµg/mÂ³",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly observations")

    monthly_plot = monthly.copy()
    monthly_plot["Date"] = pd.to_datetime(
        monthly_plot["Year_Month"] + "-01",
        errors="coerce",
    )

    monthly_pollutant_column = {
        "NO2": "NO2_Average",
        "PM2.5": "PM2_5_Average",
        "PM10": "PM10_Average",
    }[pollutant]

    monthly_plot = monthly_plot.dropna(
        subset=["Date", monthly_pollutant_column]
    )

    fig_month = px.line(
        monthly_plot,
        x="Date",
        y=monthly_pollutant_column,
        markers=True,
        title=f"Monthly {pollutant} measurements",
        labels={monthly_pollutant_column: f"{pollutant} (Âµg/mÂ³)"},
    )

    st.plotly_chart(fig_month, use_container_width=True)

    st.caption(
        "Missing observations are left missing. The dashboard does not replace "
        "missing months with zero."
    )

# ============================================================
# ZONE RISK
# ============================================================
with tab_zones:
    st.subheader("Four-zone air-risk comparison")

    st.write(
        "The four zones are development-plan screening zones. "
        "They are not separate monitoring stations."
    )

    zone_choice = st.selectbox(
        "Choose zone for detailed calculation",
        list(ZONE_MULTIPLIERS.keys()),
        key="zone_detail",
    )

    z = year_zones[year_zones["Zone"] == zone_choice].iloc[0]

    a, b, c = st.columns(3)

    with a:
        st.metric(
            "Estimated NOâ‚‚",
            f"{z['NO2_Calculated']:.2f} Âµg/mÂ³",
            f"{z['NO2_WHO_Percent']:.1f}% of WHO guideline",
        )

    with b:
        st.metric(
            "Estimated PMâ‚‚.â‚…",
            f"{z['PM2.5_Calculated']:.2f} Âµg/mÂ³",
            f"{z['PM2.5_WHO_Percent']:.1f}% of WHO guideline",
        )

    with c:
        st.metric(
            "Air-risk score",
            f"{z['Air_Risk_Score']:.1f}",
            z["Air_Risk_Class"],
        )

    st.markdown(
        f""" **Calculation chain** Station concentration â†’ Ã— **{z['Air_Multiplier']:.2f} zone multiplier** â†’ estimated zone concentration â†’ Ã· WHO guideline Ã— 100 â†’ pollutant percentages â†’ average NOâ‚‚ and PMâ‚‚.â‚… percentages â†’ **air-risk score** â†’ risk class. """
    )

    detail = pd.DataFrame(
        {
            "Measure": [
                "Station NOâ‚‚",
                "Zone NOâ‚‚",
                "NOâ‚‚ WHO %",
                "Station PMâ‚‚.â‚…",
                "Zone PMâ‚‚.â‚…",
                "PMâ‚‚.â‚… WHO %",
                "Station PMâ‚â‚€",
                "Zone PMâ‚â‚€",
                "PMâ‚â‚€ WHO %",
                "Air-risk score",
                "Risk class",
            ],
            "Value": [
                z["NO2_Station"],
                z["NO2_Calculated"],
                z["NO2_WHO_Percent"],
                z["PM2.5_Station"],
                z["PM2.5_Calculated"],
                z["PM2.5_WHO_Percent"],
                z["PM10_Station"],
                z["PM10_Calculated"],
                z["PM10_WHO_Percent"],
                z["Air_Risk_Score"],
                z["Air_Risk_Class"],
            ],
        }
    )

    st.dataframe(detail.round(2), use_container_width=True, hide_index=True)

# ============================================================
# DATA & CALCULATIONS
# ============================================================
with tab_data:
    st.subheader("Data quality and calculation audit")

    st.write(
        "This section is deliberately included so the dashboard can be checked "
        "rather than treated as a black box."
    )

    st.write("### Monthly source data")
    st.dataframe(monthly, use_container_width=True, hide_index=True)

    st.write("### Recalculated station annual values")
    st.dataframe(
        station_calc.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Zone calculations")
    st.dataframe(
        zone_calc.round(3),
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Data completeness")

    completeness = pd.DataFrame(
        {
            "Year": station_calc["Year"],
            "NO2 valid observations": station_calc["NO2_Valid_Observations"],
            "PM2.5 valid observations": station_calc[
                "PM2.5_Valid_Observations"
            ],
            "PM10 valid observations": station_calc[
                "PM10_Valid_Observations"
            ],
        }
    )

    st.dataframe(
        completeness,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Valid observation counts come from the source workbook. "
        "Unavailable measurements are not converted to zero."
    )

# ============================================================
# METHODOLOGY
# ============================================================
with tab_method:
    st.subheader("Methodology and limitations")

    st.write("### 1. Station annual calculation")
    st.code(
        "Annual mean = Î£(monthly mean Ã— valid observations) / Î£(valid observations)",
        language="text",
    )

    st.write("### 2. Zone estimate")
    st.code(
        "Estimated zone concentration = station concentration Ã— zone multiplier",
        language="text",
    )

    st.write("### 3. WHO comparison")
    st.code(
        "WHO percentage = estimated concentration / WHO guideline Ã— 100",
        language="text",
    )

    st.write("### 4. Air-risk score")
    st.code(
        "Air-risk score = (NOâ‚‚ WHO % + PMâ‚‚.â‚… WHO %) / 2",
        language="text",
    )

    st.write("### 5. Screening classification")
    st.code(
        """if score <= 100: LOW elif score <= 150: MODERATE else: HIGH""",
        language="python",
    )

    st.write("### Project zone multipliers")
    multiplier_table = pd.DataFrame(
        {
            "Zone": list(ZONE_MULTIPLIERS.keys()),
            "Multiplier": list(ZONE_MULTIPLIERS.values()),
            "Interpretation": [
                "Project screening assumption",
                "Project screening assumption",
                "Project screening assumption",
                "Project screening assumption",
            ],
        }
    )

    st.dataframe(
        multiplier_table,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "The zone multipliers are project screening assumptions, not official "
        "DEFRA or Birmingham City Council constants. The dashboard therefore "
        "presents zone values as estimates and relative screening results, "
        "not direct measurements or legal compliance determinations."
    )

    st.write("### 2026")
    st.info(
        "If 2026 contains only year-to-date observations, interpret its annual "
        "values as provisional/YTD rather than as a completed calendar-year average."
    )

# ============================================================
# FOOTER
# ============================================================
st.divider()

st.caption(
    "Ladywood Air-Risk Dashboard | Built from the supplied monitoring workbook | "
    "No synthetic sensor stream is generated."
)
