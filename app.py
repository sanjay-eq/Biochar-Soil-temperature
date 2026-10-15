import base64
import json
import math
import os
from datetime import date, timedelta
from pathlib import Path

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

# --------------------------------------------------------------------------- #
# Dataset configuration
# Each depth entry: (EE band name, display label, output column name)
# --------------------------------------------------------------------------- #
DATASETS = {
    "ERA5-Land": {
        "id": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "source": "ERA5-Land (ECMWF)",
        "caption": "Reanalysis, daily, ~11 km, 1950 – present",
        "resolution": "~0.1° (≈9–11 km)",
        "scale": 11132,
        "daily": True,              # already daily images
        "max_days_per_chunk": None,  # no extra cap
        "start": date(1950, 1, 1),
        "lookback_days": 365,
        "slug": "era5land",
        "depths": [
            ("soil_temperature_level_1", "Level 1: 0–7 cm", "soil_temp_0_7cm_C"),
            ("soil_temperature_level_2", "Level 2: 7–28 cm", "soil_temp_7_28cm_C"),
            ("soil_temperature_level_3", "Level 3: 28–100 cm", "soil_temp_28_100cm_C"),
            ("soil_temperature_level_4", "Level 4: 100–289 cm", "soil_temp_100_289cm_C"),
        ],
    },
    "SMAP L4": {
        "id": "NASA/SMAP/SPL4SMGP/008",
        "source": "SMAP L4 (NASA)",
        "caption": "Model + satellite assimilation, 3-hourly → daily, ~9–11 km, 2015 – present",
        "resolution": "9 km EASE-Grid (served by Earth Engine at ~11 km)",
        "scale": 11000,
        "daily": False,             # 3-hourly images, aggregated to daily means
        "max_days_per_chunk": 180,  # keeps 3-hourly → daily computation within EE limits
        "start": date(2015, 3, 31),
        "lookback_days": 180,
        "slug": "smap_l4",
        # Depth ranges follow the Catchment land-model soil temperature layers (approximate)
        "depths": [
            ("soil_temp_layer1", "Layer 1: 0–10 cm", "soil_temp_L1_0_10cm_C"),
            ("soil_temp_layer2", "Layer 2: 10–29 cm", "soil_temp_L2_10_29cm_C"),
            ("soil_temp_layer3", "Layer 3: 29–68 cm", "soil_temp_L3_29_68cm_C"),
            ("soil_temp_layer4", "Layer 4: 68–144 cm", "soil_temp_L4_68_144cm_C"),
            ("soil_temp_layer5", "Layer 5: 144–295 cm", "soil_temp_L5_144_295cm_C"),
            ("soil_temp_layer6", "Layer 6: 295–1295 cm", "soil_temp_L6_295_1295cm_C"),
        ],
    },
}


def bands_of(ds_key):
    return [d[0] for d in DATASETS[ds_key]["depths"]]


def band_label(ds_key):
    return {d[0]: d[1] for d in DATASETS[ds_key]["depths"]}


def band_column(ds_key):
    return {d[0]: d[2] for d in DATASETS[ds_key]["depths"]}


def temp_cols(ds_key):
    return [d[2] for d in DATASETS[ds_key]["depths"]]


def column_label(ds_key):
    return {d[2]: d[1] for d in DATASETS[ds_key]["depths"]}


PALETTE = [
    "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
    "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
]

MAX_POINTS = 500
MAX_POINT_DAYS = 100_000
FEATURES_PER_REQUEST = 4000

LAT_ALIASES = ["lat", "latitude", "y"]
LON_ALIASES = ["lon", "lng", "long", "longitude", "x"]
ID_ALIASES = ["id", "point_id", "name", "site", "station"]


# --------------------------------------------------------------------------- #
# Earth Engine setup
# --------------------------------------------------------------------------- #
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
        creds = ee.ServiceAccountCredentials(sa["client_email"], key_data=json.dumps(sa))
        ee.Initialize(creds, project=sa["project_id"])
    else:
        ee.Initialize(project=os.environ.get("EE_PROJECT"))
    return True


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def latest_available_date(ds_key: str) -> date:
    """Most recent day available in the chosen collection."""
    cfg = DATASETS[ds_key]
    today = ee.Date(date.today().isoformat())
    recent = ee.ImageCollection(cfg["id"]).filterDate(
        today.advance(-cfg["lookback_days"], "day"), today.advance(1, "day")
    )
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

    dup = pts["point_id"].duplicated(keep=False)
    if dup.any():
        suffix = pts.loc[dup].groupby("point_id").cumcount().add(1).astype(str)
        pts.loc[dup, "point_id"] = pts.loc[dup, "point_id"] + "_" + suffix

    return pts.reset_index(drop=True), n_before - len(pts)


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #
def daily_collection(ds_key: str, start: str, end_exclusive: str) -> ee.ImageCollection:
    """Daily images with soil-temperature bands for [start, end_exclusive)."""
    cfg = DATASETS[ds_key]
    bands = bands_of(ds_key)
    coll = ee.ImageCollection(cfg["id"]).filterDate(start, end_exclusive).select(bands)
    if cfg["daily"]:
        return coll

    # 3-hourly → daily mean (UTC days)
    proj = ee.Image(ee.ImageCollection(cfg["id"]).first()).select(bands[0]).projection()
    start_d = ee.Date(start)
    n_days = ee.Date(end_exclusive).difference(start_d, "day").round().int()

    def daily_mean(i):
        d = start_d.advance(i, "day")
        return (
            coll.filterDate(d, d.advance(1, "day"))
            .mean()
            .setDefaultProjection(proj)
            .set("system:time_start", d.millis())
        )

    days = ee.ImageCollection.fromImages(ee.List.sequence(0, n_days.subtract(1)).map(daily_mean))
    # Drop days with no source images (their mean has no bands)
    return days.filter(ee.Filter.listContains("system:band_names", bands[0]))


@st.cache_data(show_spinner=False)
def fetch_chunk(ds_key: str, points: tuple, start: str, end_exclusive: str) -> list:
    """Sample all depths at all points for every day in [start, end_exclusive)."""
    cfg = DATASETS[ds_key]
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"point_id": pid})
        for pid, lat, lon in points
    ])
    images = daily_collection(ds_key, start, end_exclusive)

    def sample(img):
        day = img.date().format("YYYY-MM-dd")
        samples = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=cfg["scale"])
        return samples.map(lambda f: ee.Feature(None, f.toDictionary().set("date", day)))

    result = images.map(sample).flatten()
    return [f["properties"] for f in result.getInfo()["features"]]


def fetch_soil_temperature(ds_key: str, points_df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """Fetch in date chunks so each request stays below Earth Engine's limits."""
    cfg = DATASETS[ds_key]
    points = tuple(
        (r.point_id, float(r.latitude), float(r.longitude)) for r in points_df.itertuples()
    )
    days_per_chunk = max(1, FEATURES_PER_REQUEST // len(points))
    if cfg["max_days_per_chunk"]:
        days_per_chunk = min(days_per_chunk, cfg["max_days_per_chunk"])

    chunks, cur = [], start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=days_per_chunk - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    rows = []
    label = f"Fetching {ds_key} data…"
    progress = st.progress(0.0, text=label)
    for i, (cs, ce) in enumerate(chunks, start=1):
        rows.extend(fetch_chunk(ds_key, points, cs.isoformat(), (ce + timedelta(days=1)).isoformat()))
        progress.progress(i / len(chunks), text=f"{label} ({i}/{len(chunks)})")
    progress.empty()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for band, _, col in cfg["depths"]:
        if band in df:
            df[col] = (pd.to_numeric(df[band], errors="coerce") - 273.15).round(2)  # K → °C
        else:
            df[col] = float("nan")

    df = df.merge(points_df, on="point_id", how="left")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    out_cols = ["point_id", "latitude", "longitude", "date"] + temp_cols(ds_key)
    return df[out_cols].sort_values(["point_id", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Map helpers
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def mean_tile_url(ds_key: str, band: str, start: str, end_exclusive: str, vmin: float, vmax: float) -> str:
    """Tile URL for the period-mean soil temperature (°C) of one depth."""
    cfg = DATASETS[ds_key]
    coll = ee.ImageCollection(cfg["id"])
    proj = ee.Image(coll.first()).select(band).projection()
    img = (
        coll.filterDate(start, end_exclusive)
        .select(band)
        .mean()
        .setDefaultProjection(proj)
        .subtract(273.15)
    )
    map_id = img.getMapId({"min": vmin, "max": vmax, "palette": PALETTE})
    return map_id["tile_fetcher"].url_format


def build_map(ds_key, tile_url, band, points_df, point_means, vmin, vmax):
    m = folium.Map(
        location=[points_df["latitude"].mean(), points_df["longitude"].mean()],
        zoom_start=6,
        tiles=None,
        control_scale=True,
    )
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
        attr=f"{DATASETS[ds_key]['source']} via Google Earth Engine",
        name=f"{ds_key} mean soil temperature – {band_label(ds_key)[band]}",
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
# Logo
# --------------------------------------------------------------------------- #
LOGO_PATH = Path(__file__).parent / "assets" / "eq.png"


def sidebar_logo(path: Path):
    if not path.exists():
        return
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml"}.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    st.sidebar.markdown(
        f"""
        <div style="background:#ffffff; padding:14px 18px; border-radius:12px;
                    text-align:center; margin:0 0 1.25rem 0;">
            <img src="data:{mime};base64,{b64}" alt="Logo"
                 style="max-width:100%; max-height:64px; height:auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
st.title("Soil Temperature Explorer")
st.caption("Daily soil temperature at multiple depths from ERA5-Land (ECMWF) or SMAP L4 (NASA), "
           "accessed through Google Earth Engine.")

try:
    init_ee()
except Exception as e:
    st.error(f"Earth Engine initialization failed: {e}")
    st.stop()

points_df = None
sidebar_logo(LOGO_PATH)
with st.sidebar:
    # 1. Dataset
    st.header("1. Dataset")
    ds_options = list(DATASETS)
    ds_key = st.radio(
        "Soil temperature source",
        ds_options,
        captions=[DATASETS[k]["caption"] for k in ds_options],
    )
    cfg = DATASETS[ds_key]

    try:
        latest = latest_available_date(ds_key)
    except Exception:
        latest = date.today() - timedelta(days=90)

    # 2. Location
    st.header("2. Location")
    mode = st.radio("Input type", ["Single coordinate", "Upload CSV"], horizontal=True)

    if mode == "Single coordinate":
        pid = st.text_input("Point name", "Point 1")
        lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=13.85660, format="%.5f")
        lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.35220, format="%.5f")
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

    # 3. Date range
    st.header("3. Date range")
    default_start = max(cfg["start"], latest - timedelta(days=30))
    date_range = st.date_input(
        "Start and end date",
        value=(default_start, latest),
        min_value=cfg["start"],
        max_value=latest,
        format="YYYY-MM-DD",
    )
    st.caption(f"{ds_key} available: {cfg['start']:%Y-%m-%d} to {latest:%Y-%m-%d}")

    fetch_clicked = st.button(f"Fetch {ds_key} soil temperature", type="primary", use_container_width=True)


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
                result_df = fetch_soil_temperature(ds_key, points_df, start, end)
                st.session_state["result"] = {
                    "dataset": ds_key,
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
    st.info("Choose a dataset, location and date range in the sidebar, then click **Fetch**.")
    st.stop()

res_ds = result["dataset"]
res_cfg = DATASETS[res_ds]
df = result["df"]
pts = result["points"]
start, end = result["start"], result["end"]
end_exclusive = (end + timedelta(days=1)).isoformat()

TEMP_COLS = temp_cols(res_ds)
BAND_LABEL = band_label(res_ds)
BAND_COLUMN = band_column(res_ds)
COLUMN_LABEL = column_label(res_ds)

if res_ds != ds_key:
    st.info(f"Showing your previous **{res_ds}** results. Click **Fetch {ds_key} soil temperature** "
            f"in the sidebar to load {ds_key} data.")

if df.empty:
    st.warning("No data was returned for this selection.")
    st.stop()

empty_points = df.groupby("point_id")[TEMP_COLS].apply(lambda g: g.isna().all().all())
no_data_ids = empty_points[empty_points].index.tolist()
if no_data_ids:
    shown = ", ".join(no_data_ids[:10]) + (" …" if len(no_data_ids) > 10 else "")
    st.warning(
        f"No data for: {shown}. {res_ds} covers land only, so points over water "
        "or right at the coastline can come back empty."
    )

m0, m1, m2, m3 = st.columns(4)
m0.metric("Dataset", res_ds)
m1.metric("Points", len(pts))
m2.metric("Days", (end - start).days + 1)
surface_mean = df[TEMP_COLS[0]].mean()
m3.metric(f"Mean {COLUMN_LABEL[TEMP_COLS[0]].split(': ')[-1]} temp",
          "—" if math.isnan(surface_mean) else f"{surface_mean:.1f} °C")

# Map
st.subheader(f"{res_ds} mean soil temperature, {start:%Y-%m-%d} to {end:%Y-%m-%d}")
c1, c2 = st.columns([1, 1])
band = c1.selectbox("Depth", bands_of(res_ds), format_func=lambda b: BAND_LABEL[b])

col_for_band = BAND_COLUMN[band]
data_min, data_max = df[col_for_band].min(), df[col_for_band].max()
if pd.isna(data_min) or pd.isna(data_max):
    default_range = (-10, 40)
else:
    default_range = (max(-50, math.floor(data_min) - 5), min(60, math.ceil(data_max) + 5))
vmin, vmax = c2.slider("Colour range (°C)", min_value=-50, max_value=60, value=default_range)

try:
    with st.spinner("Rendering map tiles…"):
        tile_url = mean_tile_url(res_ds, band, start.isoformat(), end_exclusive, float(vmin), float(vmax))
    point_means = df.groupby("point_id")[col_for_band].mean().to_dict()
    fmap = build_map(res_ds, tile_url, band, pts, point_means, vmin, vmax)
    st_folium(fmap, height=520, use_container_width=True, returned_objects=[])
    st.caption(
        f"The raster shows the per-pixel mean over the selected period at {res_ds}'s "
        f"{res_cfg['resolution']} resolution. Hover over a point to see its mean value."
    )
except Exception as e:
    st.error(f"Could not render the map: {e}")

# Table + download
st.subheader(f"Extracted {res_ds} soil temperature (°C)")
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
    file_name=f"{res_cfg['slug']}_soil_temp_{start:%Y%m%d}_{end:%Y%m%d}.csv",
    mime="text/csv",
)

# Time series
st.subheader("Time series by depth")
point_ids = pts["point_id"].tolist()
selected = st.selectbox("Point", point_ids) if len(point_ids) > 1 else point_ids[0]
series = df[df["point_id"] == selected].copy()
series["date"] = pd.to_datetime(series["date"])
st.line_chart(series.set_index("date")[TEMP_COLS].rename(columns=COLUMN_LABEL), y_label="°C")