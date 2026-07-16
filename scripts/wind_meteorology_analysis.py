#!/usr/bin/env python3
"""
Campbell Scientific TOA5 wind meteorology analysis.

Loads TOA5 data (plain CSV or RTF-wrapped), performs quality control,
statistical and wind-engineering analysis, and writes publication-quality
figures (300 dpi PNG) and CSV tables to wind_outputs/.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import weibull_min
from windrose import WindroseAxes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
WIND_DIR = ROOT / "data" / "wind"
OUTPUT_DIR = ROOT / "outputs" / "wind"

FIGSIZE_TS = (10, 4.5)
FIGSIZE_SQ = (8, 8)
FIGSIZE_WIDE = (12, 5)
DPI = 300

TOA5_SKIPROWS = 4
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"
UCASS_OVERLAP_SOURCES = {
    1: {"csv": "UCASS13.csv", "sep": ";", "id_col": "UCASS_ID"},
    2: {"csv": "UCASS62.csv", "sep": ",", "id_col": "UCASS_ID.1"},
    6: {"csv": "UCASS62.csv", "sep": ",", "id_col": "UCASS_ID"},
}
WIND_DATA_PATTERN = re.compile(
    r'"?(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"?,\s*'
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"(\d+)"
)

COLUMN_MAP = {
    "TIMESTAMP": "Timestamp",
    "AirTC_Avg": "AirTC_C",
    "RH": "RH_pct",
    "WindDir": "WindDir_deg",
    "WS_ms_Avg": "WS_ms",
    "BP_mbar_Avg": "BP_mbar",
}

ANALYSIS_VARS = [
    "AirTC_C",
    "RH_pct",
    "WS_ms",
    "WindDir_deg",
    "BP_mbar",
]

VAR_LABELS = {
    "AirTC_C": "Air temperature (°C)",
    "RH_pct": "Relative humidity (%)",
    "WS_ms": "Wind speed (m s$^{-1}$)",
    "WindDir_deg": "Wind direction (°)",
    "BP_mbar": "Atmospheric pressure (mbar)",
    "u_ms": "$u$ component (m s$^{-1}$)",
    "v_ms": "$v$ component (m s$^{-1}$)",
}

BEAUFORT_THRESHOLDS = [
    (0.0, 0.3, 0, "Calm"),
    (0.3, 1.6, 1, "Light air"),
    (1.6, 3.4, 2, "Light breeze"),
    (3.4, 5.5, 3, "Gentle breeze"),
    (5.5, 8.0, 4, "Moderate breeze"),
    (8.0, 10.8, 5, "Fresh breeze"),
    (10.8, 13.9, 6, "Strong breeze"),
    (13.9, 17.2, 7, "Near gale"),
    (17.2, 20.8, 8, "Gale"),
    (20.8, 24.5, 9, "Strong gale"),
    (24.5, 28.5, 10, "Storm"),
    (28.5, 32.7, 11, "Violent storm"),
    (32.7, np.inf, 12, "Hurricane"),
]

WS_FREQ_BINS = np.arange(0, 15.5, 0.5)


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_wind_file(wind_dir: Path) -> Path:
    candidates = sorted(
        list(wind_dir.glob("*.rtf"))
        + list(wind_dir.glob("*.RTF"))
        + list(wind_dir.glob("*.csv"))
        + list(wind_dir.glob("*.CSV"))
        + list(wind_dir.glob("*.dat"))
        + list(wind_dir.glob("*.DAT"))
    )
    if not candidates:
        raise FileNotFoundError(f"No TOA5 wind files found in {wind_dir}")
    return candidates[0]


def load_toa5_from_rtf(rtf_path: Path) -> pd.DataFrame:
    rtf_text = rtf_path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for raw_line in rtf_text.split("\n"):
        line = raw_line.strip().rstrip("\\")
        match = WIND_DATA_PATTERN.search(line)
        if match:
            rows.append(match.groups())

    if not rows:
        raise ValueError(f"No TOA5 data rows found in {rtf_path}")

    return pd.DataFrame(
        rows,
        columns=[
            "TIMESTAMP",
            "RECORD",
            "AirTC_Avg",
            "RH",
            "WindDir",
            "WS_ms_Avg",
            "WSDiag",
            "BP_mbar_Avg",
        ],
    )


def load_toa5_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".rtf":
        raw = load_toa5_from_rtf(path)
    else:
        # Campbell TOA5: row 0 meta, row 1 names, row 2 units, row 3 processing
        raw = pd.read_csv(path, skiprows=[0, 2, 3], header=0, low_memory=False)

    df = raw.rename(columns=COLUMN_MAP).copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    numeric_cols = ["AirTC_C", "RH_pct", "WindDir_deg", "WS_ms", "BP_mbar"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def resolve_experiment_day() -> pd.Timestamp:
    """Calendar date (UTC) of the UCASS intercomparison from common 1/2/6 overlap."""
    repo = ROOT
    master = None
    raw_cache: dict[tuple[Path, str], pd.DataFrame] = {}

    for uid, cfg in UCASS_OVERLAP_SOURCES.items():
        csv_path = repo / "data" / "ucass" / cfg["csv"]
        cache_key = (csv_path, cfg["sep"])
        if cache_key not in raw_cache:
            raw = pd.read_csv(csv_path, skiprows=4, sep=cfg["sep"], low_memory=False)
            raw["Timestamp"] = pd.to_datetime(
                raw["GPS_Date"].astype(str) + " " + raw["GPS_Time[UTC]"].astype(str),
                format=UCASS_DATE_FORMAT,
                errors="coerce",
            )
            raw_cache[cache_key] = raw.dropna(subset=["Timestamp"])

        sub = raw_cache[cache_key].loc[
            raw_cache[cache_key][cfg["id_col"]] == uid, ["Timestamp"]
        ].copy()
        if sub["Timestamp"].duplicated().any():
            sub = sub.groupby("Timestamp", as_index=False).size()[["Timestamp"]]

        master = sub if master is None else pd.merge(
            master, sub, on="Timestamp", how="inner", validate="one_to_one"
        )

    if master is None or master.empty:
        raise RuntimeError("Could not resolve UCASS experiment day from overlap data.")

    return pd.Timestamp(master["Timestamp"].min().date())


def campaign_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Experiment day and the previous calendar day (UTC), inclusive."""
    experiment = resolve_experiment_day()
    previous = experiment - pd.Timedelta(days=1)
    return previous, experiment


def filter_campaign_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep experiment day and the previous day only (exclude the day after)."""
    day_before, experiment_day = campaign_dates()
    allowed = {day_before.date(), experiment_day.date()}
    mask = df["Timestamp"].dt.date.isin(allowed)
    return df.loc[mask].reset_index(drop=True)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wd_rad = np.deg2rad(out["WindDir_deg"])
    # Meteorological convention: direction wind is coming FROM.
    out["u_ms"] = -out["WS_ms"] * np.sin(wd_rad)
    out["v_ms"] = -out["WS_ms"] * np.cos(wd_rad)
    out["hour"] = out["Timestamp"].dt.hour
    out["month"] = out["Timestamp"].dt.month
    out["date"] = out["Timestamp"].dt.date
    return out


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def circular_mean_deg(angles_deg: np.ndarray) -> float:
    valid = angles_deg[~np.isnan(angles_deg)]
    if len(valid) == 0:
        return np.nan
    rad = np.deg2rad(valid)
    mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.mod(np.rad2deg(mean_rad), 360.0))


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ANALYSIS_VARS + ["u_ms", "v_ms"]:
        n_missing = int(df[col].isna().sum())
        rows.append({
            "variable": col,
            "n_missing": n_missing,
            "pct_missing": 100.0 * n_missing / len(df) if len(df) else 0.0,
        })
    return pd.DataFrame(rows)


def duplicate_timestamp_report(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df["Timestamp"].duplicated(keep=False)
    dup_df = df.loc[dup_mask, ["Timestamp"] + ANALYSIS_VARS].copy()
    dup_df["duplicate_rank"] = dup_df.groupby("Timestamp").cumcount() + 1
    summary = (
        df.groupby("Timestamp", observed=True)
        .size()
        .reset_index(name="n_records")
    )
    summary = summary[summary["n_records"] > 1].sort_values("n_records", ascending=False)
    return dup_df, summary


def iqr_outlier_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    outlier_frames = []

    for col in ANALYSIS_VARS:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = int(mask.sum())
        summary_rows.append({
            "variable": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "n_outliers": n_outliers,
            "pct_outliers": 100.0 * n_outliers / len(df) if len(df) else 0.0,
        })
        if n_outliers:
            flagged = df.loc[mask, ["Timestamp", col]].copy()
            flagged["variable"] = col
            outlier_frames.append(flagged)

    summary = pd.DataFrame(summary_rows)
    details = (
        pd.concat(outlier_frames, ignore_index=True)
        if outlier_frames
        else pd.DataFrame(columns=["Timestamp", "value", "variable"])
    )
    return summary, details


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    percentiles = [5, 25, 50, 75, 95]

    for col in ANALYSIS_VARS:
        series = df[col].dropna().to_numpy(dtype=float)
        if len(series) == 0:
            continue
        mean = float(np.mean(series))
        std = float(np.std(series, ddof=1)) if len(series) > 1 else 0.0
        pcts = np.percentile(series, percentiles)
        rows.append({
            "variable": col,
            "n": len(series),
            "mean": mean,
            "median": float(np.median(series)),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "std": std,
            "variance": float(std ** 2),
            "coefficient_of_variation_pct": 100.0 * std / mean if mean != 0 else np.nan,
            "skewness": float(stats.skew(series, bias=False)),
            "kurtosis": float(stats.kurtosis(series, bias=False)),
            "p05": float(pcts[0]),
            "p25": float(pcts[1]),
            "p50": float(pcts[2]),
            "p75": float(pcts[3]),
            "p95": float(pcts[4]),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Wind engineering tables
# ---------------------------------------------------------------------------

def wind_speed_frequency_table(ws: np.ndarray) -> pd.DataFrame:
    counts, edges = np.histogram(ws[~np.isnan(ws)], bins=WS_FREQ_BINS)
    total = counts.sum()
    rows = []
    for i, count in enumerate(counts):
        rows.append({
            "bin_lower_ms": edges[i],
            "bin_upper_ms": edges[i + 1],
            "count": int(count),
            "frequency_pct": 100.0 * count / total if total else 0.0,
            "cumulative_pct": 100.0 * counts[: i + 1].sum() / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def beaufort_frequency_table(ws: np.ndarray) -> pd.DataFrame:
    rows = []
    valid = ws[~np.isnan(ws)]
    total = len(valid)
    for lower, upper, force, label in BEAUFORT_THRESHOLDS:
        mask = (valid >= lower) & (valid < upper)
        count = int(mask.sum())
        rows.append({
            "beaufort_force": force,
            "description": label,
            "speed_lower_ms": lower,
            "speed_upper_ms": upper if np.isfinite(upper) else np.nan,
            "count": count,
            "frequency_pct": 100.0 * count / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def fit_weibull(ws: np.ndarray) -> tuple[float, float, float]:
    valid = ws[~np.isnan(ws)]
    valid = valid[valid > 0]
    if len(valid) < 10:
        return np.nan, np.nan, np.nan
    shape, loc, scale = weibull_min.fit(valid, floc=0)
    return float(shape), float(loc), float(scale)


# ---------------------------------------------------------------------------
# Temporal aggregation
# ---------------------------------------------------------------------------

def monthly_wind_statistics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("month", observed=True)["WS_ms"]
    return grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    ).reset_index().rename(columns={"month": "month_number"})


def hourly_wind_speed_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1 Hz wind speed to hourly means (calendar UTC hours)."""
    return (
        df.set_index("Timestamp")["WS_ms"]
        .resample("h")
        .agg(mean="mean", std="std", count="count")
        .dropna(subset=["mean"])
        .reset_index()
    )


def hourly_wind_direction_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1 Hz wind direction to hourly circular means (calendar UTC hours)."""
    def _cmean(series: pd.Series) -> float:
        return circular_mean_deg(series.to_numpy(dtype=float))

    return (
        df.set_index("Timestamp")["WindDir_deg"]
        .resample("h")
        .apply(_cmean)
        .dropna()
        .reset_index()
        .rename(columns={"WindDir_deg": "WindDir_circular_mean_deg"})
    )


def daily_hourly_wind_statistics(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot_table(
            index="date",
            columns="hour",
            values="WS_ms",
            aggfunc="mean",
            observed=True,
        )
        .sort_index()
    )
    pivot.index = pivot.index.astype(str)
    pivot.columns = [f"hour_{int(c):02d}" for c in pivot.columns]
    return pivot.reset_index().rename(columns={"date": "date_utc"})


# ---------------------------------------------------------------------------
# Plotting — basic time series
# ---------------------------------------------------------------------------

def plot_timeseries(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    filename: str,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_TS)
    ax.plot(df["Timestamp"], df[column], linewidth=0.6, color="#1f4e79")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if period_start is not None and period_end is not None:
        add_overlap_period_frame(ax, period_start, period_end)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Plotting — wind analysis
# ---------------------------------------------------------------------------

def _ucass_overlap_period_bounds(wind_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive [start, end] where UCASS 1, 2, and 6 all have 1 Hz data with wind."""
    repo = ROOT
    master = None
    raw_cache: dict[tuple[Path, str], pd.DataFrame] = {}

    for uid, cfg in UCASS_OVERLAP_SOURCES.items():
        csv_path = repo / "data" / "ucass" / cfg["csv"]
        cache_key = (csv_path, cfg["sep"])
        if cache_key not in raw_cache:
            raw = pd.read_csv(csv_path, skiprows=4, sep=cfg["sep"], low_memory=False)
            raw["Timestamp"] = pd.to_datetime(
                raw["GPS_Date"].astype(str) + " " + raw["GPS_Time[UTC]"].astype(str),
                format=UCASS_DATE_FORMAT,
                errors="coerce",
            )
            raw_cache[cache_key] = raw.dropna(subset=["Timestamp"])

        sub = raw_cache[cache_key].loc[
            raw_cache[cache_key][cfg["id_col"]] == uid, ["Timestamp"]
        ].copy()
        if sub["Timestamp"].duplicated().any():
            sub = sub.groupby("Timestamp", as_index=False).size()[["Timestamp"]]

        master = sub if master is None else pd.merge(
            master, sub, on="Timestamp", how="inner", validate="one_to_one"
        )

    wind_ts = wind_df[["Timestamp"]].drop_duplicates()
    master = pd.merge(master, wind_ts, on="Timestamp", how="inner", validate="one_to_one")
    return master["Timestamp"].min(), master["Timestamp"].max()


OVERLAP_FRAME_LINEWIDTH = 0.8

TS_OVERLAP_HIGHLIGHT = frozenset({
    "wind_speed_timeseries.png",
    "wind_direction_timeseries.png",
    "relative_humidity_timeseries.png",
    "air_temperature_timeseries.png",
    "atmospheric_pressure_timeseries.png",
})


def add_overlap_period_frame(
    ax: plt.Axes,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> None:
    """Thin red frame for the UCASS overlap window (matches heatmap style)."""
    from matplotlib.transforms import blended_transform_factory

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_patch(
        Rectangle(
            (period_start, 0),
            period_end - period_start,
            1,
            transform=trans,
            linewidth=OVERLAP_FRAME_LINEWIDTH,
            edgecolor="red",
            facecolor="none",
            zorder=5,
            clip_on=False,
        )
    )


def plot_wind_rose(
    df: pd.DataFrame,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path:
    if period_start is None or period_end is None:
        period_start, period_end = _ucass_overlap_period_bounds(df)
    overlap = df.loc[
        (df["Timestamp"] >= period_start) & (df["Timestamp"] <= period_end)
    ]
    valid = overlap[["WindDir_deg", "WS_ms"]].dropna()
    fig = plt.figure(figsize=FIGSIZE_SQ)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(
        valid["WindDir_deg"],
        valid["WS_ms"],
        normed=True,
        opening=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_legend(
        title="Wind speed (m s$^{-1}$)",
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
    )
    ax.set_title("Wind Rose")
    fig.tight_layout()
    return save_figure(fig, "wind_rose.png")


def plot_histogram(
    data: np.ndarray,
    xlabel: str,
    title: str,
    filename: str,
    bins: int | np.ndarray = 30,
    xlim: tuple[float, float] | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.hist(data[~np.isnan(data)], bins=bins, color="#2c6e8f", edgecolor="white", alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    return save_figure(fig, filename)


def plot_wind_direction_polar(df: pd.DataFrame) -> Path:
    valid = df["WindDir_deg"].dropna().to_numpy(dtype=float)
    fig = plt.figure(figsize=FIGSIZE_SQ)
    ax = fig.add_subplot(111, projection="polar")
    bins = np.linspace(0, 2 * np.pi, 37)
    theta = np.deg2rad(valid)
    ax.hist(theta, bins=bins, color="#3a7d44", alpha=0.85, edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Wind Direction Distribution (Polar)", pad=20)
    fig.tight_layout()
    return save_figure(fig, "wind_direction_polar.png")


def plot_diurnal_wind_speed(
    hourly_ts: pd.DataFrame,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_TS)
    ax.plot(
        hourly_ts["Timestamp"],
        hourly_ts["mean"],
        marker="o",
        markersize=3,
        linewidth=1.0,
        color="#1f4e79",
        label="Hourly mean",
    )
    ax.fill_between(
        hourly_ts["Timestamp"],
        hourly_ts["mean"] - hourly_ts["std"],
        hourly_ts["mean"] + hourly_ts["std"],
        color="#1f4e79",
        alpha=0.2,
        label="±1 SD",
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Wind speed (m s$^{-1}$)")
    ax.set_title("Hourly Mean Wind Speed")
    ax.legend()
    if period_start is not None and period_end is not None:
        add_overlap_period_frame(ax, period_start, period_end)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, "diurnal_wind_speed.png")


def plot_diurnal_wind_direction(
    hourly_ts: pd.DataFrame,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE_TS)
    ax.plot(
        hourly_ts["Timestamp"],
        hourly_ts["WindDir_circular_mean_deg"],
        marker="o",
        markersize=3,
        linewidth=1.0,
        color="#3a7d44",
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Circular mean wind direction (°)")
    ax.set_title("Hourly Mean Wind Direction")
    ax.set_ylim(0, 360)
    if period_start is not None and period_end is not None:
        add_overlap_period_frame(ax, period_start, period_end)
    fig.autofmt_xdate()
    fig.tight_layout()
    return save_figure(fig, "diurnal_wind_direction.png")


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_idx = sorted(indices)
    runs: list[tuple[int, int]] = []
    run_start = run_end = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx == run_end + 1:
            run_end = idx
        else:
            runs.append((run_start, run_end))
            run_start = run_end = idx
    runs.append((run_start, run_end))
    return runs


def _heatmap_overlap_rectangles(
    daily_hourly: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> list[tuple[int, int, int]]:
    """Return (row_idx, col_min, col_max) for UCASS overlap hour-cells in the heatmap."""
    value_cols = [c for c in daily_hourly.columns if c.startswith("hour_")]
    rectangles: list[tuple[int, int, int]] = []

    for row_idx, date_str in enumerate(daily_hourly["date_utc"].astype(str)):
        cols_in_overlap: list[int] = []
        for col_idx, col in enumerate(value_cols):
            hour = int(col.replace("hour_", ""))
            hour_start = pd.Timestamp(f"{date_str} {hour:02d}:00:00")
            hour_end = hour_start + pd.Timedelta(hours=1)
            if hour_start < period_end and hour_end > period_start:
                cols_in_overlap.append(col_idx)

        for col_min, col_max in _contiguous_runs(cols_in_overlap):
            rectangles.append((row_idx, col_min, col_max))

    return rectangles


def plot_monthly_hourly_heatmap(
    daily_hourly: pd.DataFrame,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path | None:
    value_cols = [c for c in daily_hourly.columns if c.startswith("hour_")]
    if not value_cols:
        return None
    matrix = daily_hourly[value_cols].to_numpy(dtype=float)
    if matrix.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, 0.45 * len(daily_hourly) + 2)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", origin="upper")
    ax.set_xticks(range(len(value_cols)))
    ax.set_xticklabels([c.replace("hour_", "") for c in value_cols], rotation=0)
    ax.set_yticks(range(len(daily_hourly)))
    ax.set_yticklabels(daily_hourly["date_utc"])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Date (UTC)")
    ax.set_title("Mean Wind Speed by Date and Hour")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Wind speed (m s$^{-1}$)")

    if period_start is not None and period_end is not None:
        for row_idx, col_min, col_max in _heatmap_overlap_rectangles(
            daily_hourly, period_start, period_end
        ):
            ax.add_patch(
                Rectangle(
                    (col_min - 0.5, row_idx - 0.5),
                    col_max - col_min + 1,
                    1,
                    linewidth=0.8,
                    edgecolor="red",
                    facecolor="none",
                    zorder=5,
                )
            )

    fig.tight_layout()
    return save_figure(fig, "daily_hourly_wind_speed_heatmap.png")


def plot_monthly_wind_speed(monthly_stats: pd.DataFrame) -> Path | None:
    if len(monthly_stats) < 2:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        monthly_stats["month_number"].astype(str),
        monthly_stats["mean"],
        yerr=monthly_stats["std"],
        color="#2c6e8f",
        capsize=4,
        edgecolor="white",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean wind speed (m s$^{-1}$)")
    ax.set_title("Monthly Mean Wind Speed")
    fig.tight_layout()
    return save_figure(fig, "monthly_wind_speed.png")


# ---------------------------------------------------------------------------
# Plotting — correlation
# ---------------------------------------------------------------------------

def correlation_variables(df: pd.DataFrame) -> list[str]:
    """Variables with non-zero variance (required for Pearson r)."""
    return [
        col for col in ANALYSIS_VARS
        if df[col].std(skipna=True) > 0
    ]


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    period_start: pd.Timestamp | None = None,
    period_end: pd.Timestamp | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [VAR_LABELS.get(c, c) for c in corr.columns]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    title = "Pearson Correlation Matrix"
    if period_start is not None and period_end is not None:
        title += f"\n{period_start} – {period_end} UTC"
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson $r$")
    fig.tight_layout()
    return save_figure(fig, "correlation_heatmap.png")


# ---------------------------------------------------------------------------
# Plotting — Weibull
# ---------------------------------------------------------------------------

def plot_weibull_fit(ws: np.ndarray, shape: float, scale: float) -> Path:
    valid = ws[~np.isnan(ws)]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    counts, edges, _ = ax.hist(
        valid,
        bins=30,
        density=True,
        color="#7f9db8",
        edgecolor="white",
        alpha=0.85,
        label="Observed",
    )
    x = np.linspace(0, valid.max() * 1.05, 400)
    pdf = weibull_min.pdf(x, shape, loc=0, scale=scale)
    ax.plot(x, pdf, color="#b23a48", linewidth=2.0, label=f"Weibull fit ($k$={shape:.3f}, $c$={scale:.3f} m s$^{{-1}}$)")
    ax.set_xlabel("Wind speed (m s$^{-1}$)")
    ax.set_ylabel("Probability density")
    ax.set_title("Wind Speed Distribution with Weibull Fit")
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, "wind_speed_weibull_fit.png")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_analysis(wind_file: Path | None = None) -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = wind_file or discover_wind_file(WIND_DIR)
    print(f"Loading: {source}")
    df = load_toa5_file(source)
    df = add_derived_columns(df)
    df = filter_campaign_dates(df)
    if df.empty:
        day_before, experiment_day = campaign_dates()
        raise ValueError(
            "No wind records for meteorology window "
            f"{day_before.date()} and {experiment_day.date()} (experiment day + previous day)"
        )
    print(f"Records: {len(df):,}")
    day_before, experiment_day = campaign_dates()
    print(
        f"Meteorology window: {day_before.date()} (previous) & {experiment_day.date()} (experiment) "
        f"({df['Timestamp'].min()} -> {df['Timestamp'].max()} UTC)"
    )

    overlap_start, overlap_end = _ucass_overlap_period_bounds(df)
    overlap_df = df.loc[
        (df["Timestamp"] >= overlap_start) & (df["Timestamp"] <= overlap_end)
    ]
    print(
        f"UCASS overlap period: {overlap_start} -> {overlap_end} UTC "
        f"({len(overlap_df):,} records)"
    )

    ws = df["WS_ms"].to_numpy(dtype=float)
    ws_overlap = overlap_df["WS_ms"].to_numpy(dtype=float)
    saved_figures: list[Path] = []
    saved_tables: list[Path] = []

    # --- Quality control ---
    missing = missing_value_report(df)
    saved_tables.append(save_csv(missing, "missing_value_report.csv"))

    dup_details, dup_summary = duplicate_timestamp_report(df)
    saved_tables.append(save_csv(dup_details, "duplicate_timestamps.csv"))
    saved_tables.append(save_csv(dup_summary, "duplicate_timestamp_summary.csv"))

    outlier_summary, outlier_details = iqr_outlier_report(df)
    saved_tables.append(save_csv(outlier_summary, "outlier_summary_iqr.csv"))
    saved_tables.append(save_csv(outlier_details, "outlier_details_iqr.csv"))

    # --- Descriptive statistics ---
    desc = descriptive_statistics(df)
    saved_tables.append(save_csv(desc, "descriptive_statistics.csv"))

    # --- Temporal statistics ---
    monthly_stats = monthly_wind_statistics(df)
    hourly_ws_ts = hourly_wind_speed_timeseries(df)
    hourly_dir_ts = hourly_wind_direction_timeseries(df)
    daily_hourly = daily_hourly_wind_statistics(df)
    saved_tables.append(save_csv(monthly_stats, "monthly_wind_speed_statistics.csv"))
    saved_tables.append(save_csv(hourly_ws_ts, "hourly_wind_speed_statistics.csv"))
    saved_tables.append(save_csv(daily_hourly, "daily_hourly_wind_speed_statistics.csv"))
    saved_tables.append(save_csv(hourly_dir_ts, "hourly_circular_mean_wind_direction.csv"))

    # --- Wind engineering tables ---
    ws_freq = wind_speed_frequency_table(ws)
    saved_tables.append(save_csv(ws_freq, "wind_speed_frequency_table.csv"))

    beaufort = beaufort_frequency_table(ws)
    saved_tables.append(save_csv(beaufort, "beaufort_frequency_table.csv"))

    shape, loc, scale = fit_weibull(ws_overlap)
    weibull_params = pd.DataFrame([{
        "shape_k": shape,
        "scale_c_ms": scale,
        "location": loc,
        "period_start_utc": overlap_start,
        "period_end_utc": overlap_end,
        "n_samples": int(np.sum(~np.isnan(ws_overlap))),
    }])
    saved_tables.append(save_csv(weibull_params, "weibull_fit_parameters.csv"))

    corr_vars = correlation_variables(overlap_df)
    corr = overlap_df[corr_vars].corr(method="pearson")
    saved_tables.append(save_csv(corr.reset_index().rename(columns={"index": "variable"}), "correlation_matrix.csv"))

    # --- Basic time series (1–5) ---
    ts_specs = [
        ("WS_ms", VAR_LABELS["WS_ms"], "Wind Speed Time Series", "wind_speed_timeseries.png"),
        ("WindDir_deg", VAR_LABELS["WindDir_deg"], "Wind Direction Time Series", "wind_direction_timeseries.png"),
        ("AirTC_C", VAR_LABELS["AirTC_C"], "Air Temperature Time Series", "air_temperature_timeseries.png"),
        ("RH_pct", VAR_LABELS["RH_pct"], "Relative Humidity Time Series", "relative_humidity_timeseries.png"),
        ("BP_mbar", VAR_LABELS["BP_mbar"], "Atmospheric Pressure Time Series", "atmospheric_pressure_timeseries.png"),
    ]
    for col, ylabel, title, fname in ts_specs:
        highlight = overlap_start if fname in TS_OVERLAP_HIGHLIGHT else None
        highlight_end = overlap_end if fname in TS_OVERLAP_HIGHLIGHT else None
        saved_figures.append(
            plot_timeseries(df, col, ylabel, title, fname, highlight, highlight_end)
        )

    # --- Wind analysis (6–16) ---
    saved_figures.append(plot_wind_rose(df, overlap_start, overlap_end))
    saved_figures.append(plot_histogram(ws_overlap, "Wind speed (m s$^{-1}$)", "Wind Speed Histogram", "wind_speed_histogram.png", bins=30))
    saved_figures.append(plot_histogram(
        overlap_df["WindDir_deg"].to_numpy(dtype=float),
        "Wind direction (°)",
        "Wind Direction Histogram",
        "wind_direction_histogram.png",
        bins=np.arange(0, 361, 10),
        xlim=(0, 360),
    ))
    saved_figures.append(plot_wind_direction_polar(df))
    monthly_fig = plot_monthly_wind_speed(monthly_stats)
    if monthly_fig:
        saved_figures.append(monthly_fig)
    heatmap_fig = plot_monthly_hourly_heatmap(daily_hourly, overlap_start, overlap_end)
    if heatmap_fig:
        saved_figures.append(heatmap_fig)
    saved_figures.append(plot_diurnal_wind_speed(hourly_ws_ts, overlap_start, overlap_end))
    saved_figures.append(plot_diurnal_wind_direction(hourly_dir_ts, overlap_start, overlap_end))

    # --- Correlation ---
    saved_figures.append(plot_correlation_heatmap(corr, overlap_start, overlap_end))

    # --- Weibull (23) ---
    if np.isfinite(shape) and np.isfinite(scale):
        saved_figures.append(plot_weibull_fit(ws_overlap, shape, scale))

    # --- Vector components (24) ---
    saved_figures.append(plot_timeseries(
        df, "u_ms", VAR_LABELS["u_ms"], "Zonal Wind Component ($u$)",
        "u_component_timeseries.png", overlap_start, overlap_end,
    ))
    saved_figures.append(plot_timeseries(
        df, "v_ms", VAR_LABELS["v_ms"], "Meridional Wind Component ($v$)",
        "v_component_timeseries.png", overlap_start, overlap_end,
    ))

    vector_ts = df[["Timestamp", "u_ms", "v_ms"]].copy()
    saved_tables.append(save_csv(vector_ts, "wind_vector_components.csv"))

    # --- Summary ---
    print("\nQuality control:")
    print(f"  Missing values (any core variable): {int(missing['n_missing'].sum())} cell entries")
    print(f"  Duplicate timestamps: {len(dup_summary)}")
    print(f"  IQR outliers (all variables): {int(outlier_summary['n_outliers'].sum())}")
    if np.isfinite(shape):
        print(f"\nWeibull fit: k = {shape:.3f}, c = {scale:.3f} m/s")

    print(f"\nSaved {len(saved_figures)} figures and {len(saved_tables)} tables to {OUTPUT_DIR}/")
    for path in sorted(saved_figures + saved_tables):
        print(f"  {path.name}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        run_analysis()
