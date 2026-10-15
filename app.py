import json
import math
import os
from datetime import date, timedelta

import ee
import folium
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Soil Temperature Explorer",
    layout="wide",
)

DATASET = "ECMWF/ERA5_LAND/DAILY_AGGR"
SCALE_M = 11132  # native ERA5-Land grid (~0.1°)

# (EE band name, display label, output column name)
DEPTHS = [
    ("soil_temperature_level_1", "Level 1: 0–7 cm", "soil_temp_0_7cm_C"),
    ("soil_temperature_level_2", "Level 2: 7–28 cm", "soil_temp_7_28cm_C"),
    ("soil_temperature_level_3", "Level 3: 28–100 cm", "soil_temp_28_100cm_C"),
    ("soil_temperature_level_4", "Level 4: 100–289 cm", "soil_temp_100_289cm_C"),
]
BAND_NAMES = [d[0] for d in DEPTHS]
BAND_LABEL = {d[0]: d[1] for d in DEPTHS}
BAND_COLUMN = {d[0]: d[2] for d in DEPTHS}
TEMP_COLS = [d[2] for d in DEPTHS]
COLUMN_LABEL = {d[2]: d[1] for d in DEPTHS}

PALETTE = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]

MAX_POINTS = 500
MAX_POINT_DAYS = 100_000
FEATURES_PER_REQUEST = 4000  # stay under Earth Engine's 5000-element getInfo limit

LAT_ALIASES = ["lat", "latitude", "y"]
LON_ALIASES = ["lon", "lng", "long", "longitude", "x"]
ID_ALIASES = ["id", "point_id", "name", "site", "station"]


def _service_account_info():
    """Return the service-account dict from Streamlit secrets, or None."""
    try:
        return dict(st.secrets["gcp_service_account"])
    except Exception:
        return None


@st.cache_resource(show_spinner="Connecting to Earth Engine…")
def init_ee():
    sa = _service_account_info()
    if sa:
        # Cloud deployment: authenticate silently with the service account
        creds = ee.ServiceAccountCredentials(sa["client_email"], key_data=json.dumps(sa))
        ee.Initialize(creds, project=sa["project_id"])
    else:
        # Local development: uses credentials saved by `earthengine authenticate`
        ee.Initialize(project=os.environ.get("EE_PROJECT"))
    return True


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def latest_available_date() -> date:
    """Most recent day available in the ERA5-Land daily collection."""
    today = ee.Date(date.today().isoformat())
    recent = ee.ImageCollection(DATASET).filterDate(today.advance(-1, "year"), today.advance(1, "day"))
    ms = recent.aggregate_max("system:time_start").getInfo()
    if ms is None:
        return date.today() - timedelta(days=90)
    return pd.to_datetime(ms, unit="ms").date()


# --------------------------------------------------------------------------- #
# Input helpers
# --------------------------------------------------------------------------- #
def guess_column(columns, aliases):
    lookup = {str(c).lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def clean_points(raw: pd.DataFrame, lat_col, lon_col, id_col):
    """Build a tidy points table and drop invalid coordinates."""
    pts = pd.DataFrame({
        "point_id": raw[id_col].astype(str).str.strip() if id_col else [f"P{i + 1}" for i in range(len(raw))],
        "latitude": pd.to_numeric(raw[lat_col], errors="coerce"),
        "longitude": pd.to_numeric(raw[lon_col], errors="coerce"),
    })
    n_before = len(pts)
    pts = pts.dropna(subset=["latitude", "longitude"])
    pts = pts[pts["latitude"].between(-90, 90) & pts["longitude"].between(-180, 180)].copy()

    # Make point IDs unique so results can be joined back reliably
    dup = pts["point_id"].duplicated(keep=False)
    if dup.any():
        suffix = pts.loc[dup].groupby("point_id").cumcount().add(1).astype(str)
        pts.loc[dup, "point_id"] = pts.loc[dup, "point_id"] + "_" + suffix

    return pts.reset_index(drop=True), n_before - len(pts)


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def fetch_chunk(points: tuple, start: str, end_exclusive: str) -> list:
    """Sample all bands at all points for every day in [start, end_exclusive)."""
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"point_id": pid})
        for pid, lat, lon in points
    ])
    images = ee.ImageCollection(DATASET).filterDate(start, end_exclusive).select(BAND_NAMES)

    def sample(img):
        day = img.date().format("YYYY-MM-dd")
        samples = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=SCALE_M)
        # Drop geometry to keep the response small
        return samples.map(lambda f: ee.Feature(None, f.toDictionary().set("date", day)))

    result = images.map(sample).flatten()
    return [f["properties"] for f in result.getInfo()["features"]]


def fetch_soil_temperature(points_df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Fetch in date chunks so each request stays below Earth Engine's limits."""
    points = tuple(
        (r.point_id, float(r.latitude), float(r.longitude)) for r in points_df.itertuples()
    )
    days_per_chunk = max(1, FEATURES_PER_REQUEST // len(points))

    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days_per_chunk - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    rows = []
    progress = st.progress(0.0, text="Fetching ERA5-Land data…")
    for i, (cs, ce) in enumerate(chunks, start=1):
        rows.extend(fetch_chunk(points, cs.isoformat(), (ce + timedelta(days=1)).isoformat()))
        progress.progress(i / len(chunks), text=f"Fetching ERA5-Land data… ({i}/{len(chunks)})")
    progress.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for band, _, col in DEPTHS:
        if band in df:
            df[col] = (pd.to_numeric(df[band], errors="coerce") - 273.15).round(2)  # K → °C
        else:
            df[col] = float("nan")

    df = df.merge(points_df, on="point_id", how="left")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out_cols = ["point_id", "latitude", "longitude", "date"] + TEMP_COLS
    return df[out_cols].sort_values(["point_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Map helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def mean_tile_url(band: str, start: str, end_exclusive: str, vmin: float, vmax: float) -> str:
    """Tile URL for the period-mean soil temperature (°C) of one depth."""
    img = (
        ee.ImageCollection(DATASET)
        .filterDate(start, end_exclusive)
        .select(band)
        .mean()
        .subtract(273.15)
    )
    map_id = img.getMapId({"min": vmin, "max": vmax, "palette": PALETTE})
    return map_id["tile_fetcher"].url_format



def build_map(tile_url, band, points_df, point_means, vmin, vmax):
    m = folium.Map(
        location=[points_df["latitude"].mean(), points_df["longitude"].mean()],
        zoom_start=6,
        tiles=None,
        control_scale=True,
    )
    # Key-free basemaps (the first one added is shown by default)
    folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles Esri: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="Esri Satellite",
            overlay=False,
        ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles — Esri, HERE, Garmin, USGS, and other contributors",
        name="Esri Topographic",
        overlay=False,
    ).add_to(m)
    folium.TileLayer(
        tiles=tile_url,
        attr="ERA5-Land (ECMWF) via Google Earth Engine",
        name=f"Mean soil temperature – {BAND_LABEL[band]}",
        overlay=True,
        opacity=0.75,
    ).add_to(m)
 
    markers = folium.FeatureGroup(name="Points")
    for r in points_df.itertuples():
        mean_val = point_means.get(r.point_id)
        value_txt = f"{mean_val:.2f} °C" if mean_val is not None and not math.isnan(mean_val) else "no data"
        folium.CircleMarker(
            location=[r.latitude, r.longitude],
            radius=3,
            color="#111111",
            weight=1,
            fill=True,
            fill_color="#d32727",
            fill_opacity=0.9,
            tooltip=f"<b>{r.point_id}</b><br>{r.latitude:.4f}, {r.longitude:.4f}<br>Mean: {value_txt}",
        ).add_to(markers)
    markers.add_to(m)
 
    if len(points_df) > 1:
        m.fit_bounds(
            [
                [points_df["latitude"].min(), points_df["longitude"].min()],
                [points_df["latitude"].max(), points_df["longitude"].max()],
            ],
            padding=(30, 30),
        )
 
    LinearColormap(PALETTE, vmin=vmin, vmax=vmax, caption="Mean soil temperature (°C)").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("ERA5-Land Soil Temperature Explorer")
st.caption("Daily mean soil temperature at four depths from ERA5-Land (ECMWF), accessed through Google Earth Engine.")

try:
    init_ee()
except Exception as e:
    st.error(f"Earth Engine initialization failed: {e}")
    st.stop()

try:
    latest = latest_available_date()
except Exception:
    latest = date.today() - timedelta(days=90)

# ----------------------------- Sidebar inputs ------------------------------ #
points_df = None
with st.sidebar:
    st.header("1. Location")
    mode = st.radio("Input type", ["Single coordinate", "Upload CSV"], horizontal=True)

    if mode == "Single coordinate":
        pid = st.text_input("Point name", "Point 1")
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=48.85660, format="%.5f")
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=2.35220, format="%.5f")
        points_df = pd.DataFrame({
            "point_id": [pid.strip() or "Point 1"],
            "latitude": [lat],
            "longitude": [lon],
        })
    else:
        upload = st.file_uploader(
            "CSV with coordinates",
            type="csv",
            help="Needs latitude and longitude columns in decimal degrees (WGS84). "
                 "An optional id/name column labels each point.",
        )
        st.download_button(
            "Download sample CSV",
            data="id,lat,lon\nsite_a,48.8566,2.3522\nsite_b,52.5200,13.4050\n",
            file_name="sample_points.csv",
            mime="text/csv",
        )
        if upload is not None:
            try:
                raw = pd.read_csv(upload)
            except Exception as e:
                st.error(f"Could not read the CSV: {e}")
                raw = None

            if raw is not None and raw.empty:
                st.error("The uploaded CSV is empty.")
            elif raw is not None:
                cols = list(raw.columns)
                lat_guess = guess_column(cols, LAT_ALIASES)
                lon_guess = guess_column(cols, LON_ALIASES)
                id_guess = guess_column(cols, ID_ALIASES)

                lat_col = st.selectbox("Latitude column", cols, index=cols.index(lat_guess) if lat_guess else 0)
                lon_col = st.selectbox(
                    "Longitude column", cols,
                    index=cols.index(lon_guess) if lon_guess else min(1, len(cols) - 1),
                )
                id_options = ["(auto-number)"] + cols
                id_choice = st.selectbox(
                    "ID column (optional)", id_options,
                    index=id_options.index(id_guess) if id_guess else 0,
                )

                points_df, dropped = clean_points(
                    raw, lat_col, lon_col, None if id_choice == "(auto-number)" else id_choice
                )
                if dropped:
                    st.warning(f"Skipped {dropped} row(s) with missing or invalid coordinates.")
                st.caption(f"{len(points_df)} valid point(s) loaded.")

    st.header("2. Date range")
    date_range = st.date_input(
        "Start and end date",
        value=(latest - timedelta(days=30), latest),
        min_value=date(1950, 1, 1),
        max_value=latest,
        format="YYYY-MM-DD",
    )
    st.caption(f"Latest available ERA5-Land date: {latest:%Y-%m-%d}")

    fetch_clicked = st.button("Fetch soil temperature", type="primary", use_container_width=True)

# ------------------------------ Fetch data -------------------------------- #
if fetch_clicked:
    if points_df is None or points_df.empty:
        st.sidebar.error("Enter a coordinate or upload a CSV first.")
    elif not isinstance(date_range, tuple) or len(date_range) != 2:
        st.sidebar.error("Pick both a start and an end date.")
    else:
        start, end = date_range
        n_days = (end - start).days + 1
        n_points = len(points_df)
        if n_points > MAX_POINTS:
            st.sidebar.error(f"Too many points ({n_points}). The limit is {MAX_POINTS}.")
        elif n_points * n_days > MAX_POINT_DAYS:
            st.sidebar.error(
                f"{n_points} points × {n_days} days = {n_points * n_days:,} samples, "
                f"above the {MAX_POINT_DAYS:,} limit. Shorten the date range or use fewer points."
            )
        else:
            try:
                result_df = fetch_soil_temperature(points_df, start, end)
                st.session_state["result"] = {
                    "df": result_df,
                    "points": points_df.copy(),
                    "start": start,
                    "end": end,
                }
            except ee.EEException as e:
                st.error(f"Earth Engine error: {e}")
            except Exception as e:
                st.error(f"Something went wrong while fetching data: {e}")

# ------------------------------- Results ---------------------------------- #
result = st.session_state.get("result")
if result is None:
    st.info("Choose a location and date range in the sidebar, then click **Fetch soil temperature**.")
    st.stop()

df = result["df"]
pts = result["points"]
start, end = result["start"], result["end"]
end_exclusive = (end + timedelta(days=1)).isoformat()

if df.empty:
    st.warning("No data was returned for this selection.")
    st.stop()

empty_points = df.groupby("point_id")[TEMP_COLS].apply(lambda g: g.isna().all().all())
no_data_ids = empty_points[empty_points].index.tolist()
if no_data_ids:
    shown = ", ".join(no_data_ids[:10]) + (" …" if len(no_data_ids) > 10 else "")
    st.warning(
        f"No data for: {shown}. ERA5-Land covers land only, so points over water "
        "or right at the coastline can come back empty."
    )

m1, m2, m3 = st.columns(3)
m1.metric("Points", len(pts))
m2.metric("Days", (end - start).days + 1)
surface_mean = df[TEMP_COLS[0]].mean()
m3.metric("Mean surface (0–7 cm) temp", "—" if math.isnan(surface_mean) else f"{surface_mean:.1f} °C")

# Map
st.subheader(f"Mean soil temperature, {start:%Y-%m-%d} to {end:%Y-%m-%d}")
c1, c2 = st.columns([1, 1])
band = c1.selectbox("Depth", BAND_NAMES, format_func=lambda b: BAND_LABEL[b])

col_for_band = BAND_COLUMN[band]
data_min, data_max = df[col_for_band].min(), df[col_for_band].max()
if pd.isna(data_min) or pd.isna(data_max):
    default_range = (-10, 40)
else:
    default_range = (max(-50, math.floor(data_min) - 5), min(60, math.ceil(data_max) + 5))
vmin, vmax = c2.slider("Colour range (°C)", min_value=-50, max_value=60, value=default_range)

try:
    with st.spinner("Rendering map tiles…"):
        tile_url = mean_tile_url(band, start.isoformat(), end_exclusive, float(vmin), float(vmax))
    point_means = df.groupby("point_id")[col_for_band].mean().to_dict()
    fmap = build_map(tile_url, band, pts, point_means, vmin, vmax)
    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
    st.caption(
        "The raster shows the per-pixel mean over the selected period at ERA5-Land's native "
        "~0.1° (≈9–11 km) resolution. Hover over a point to see its mean value."
    )
except Exception as e:
    st.error(f"Could not render the map: {e}")

# Table + download
st.subheader("Extracted soil temperature (°C)")
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        col: st.column_config.NumberColumn(COLUMN_LABEL[col], format="%.2f") for col in TEMP_COLS
    },
)
st.download_button(
    "⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"era5land_soil_temp_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    mime="text/csv",
)

# Time series
st.subheader("Time series by depth")
point_ids = pts["point_id"].tolist()
selected = st.selectbox("Point", point_ids) if len(point_ids) > 1 else point_ids[0]
series = df[df["point_id"] == selected].copy()
series["date"] = pd.to_datetime(series["date"])
st.line_chart(series.set_index("date")[TEMP_COLS].rename(columns=COLUMN_LABEL), y_label="°C")