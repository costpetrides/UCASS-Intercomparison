#!/usr/bin/env python3
"""
Mean dN/dlnDp PSDs stratified by wind-speed class for all reference instruments.

For each wind-speed bin, volume-weighted mean spectra are computed for
FIDAS 200, UCASS 1/2/6, and TSI OPS 3330, then plotted with the same style
as combined_FIDAS_UCASS_TSI_dN_dlnDp.png. Sample counts appear in the legend.

Outputs (per class):
  outputs/combined/wind_speed/mean_dN_dlnDp_WS_<class>.png
  outputs/combined/wind_speed/mean_dN_dlnDp_WS_<class>.csv

Dependencies: numpy, pandas, matplotlib, openpyxl
Optional: numbers-parser (if UCASS_size_bins.numbers is used for calibrations)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tsi_mean_psd import load_tsi_spectra
from plot_utils import mask_nonpositive_for_log
from fidas_utils import (
    diameter_from_column,
    geometric_bin_boundaries,
    load_fidas_excel,
    sorted_psd_columns,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "combined" / "wind_speed"

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"
FIDAS_XLSX = ROOT / "data" / "fidas" / "FIDAS200.txt"

UCASS_IDS = (1, 2, 6)
UCASS_SOURCES = {
    1: {"csv": UCASS27_CSV, "sep": ";", "id_col": "UCASS_ID", "bin_suffix": ""},
    2: {"csv": UCASS362_CSV, "sep": ",", "id_col": "UCASS_ID.1", "bin_suffix": ".1"},
    6: {"csv": UCASS362_CSV, "sep": ",", "id_col": "UCASS_ID", "bin_suffix": ""},
}
CALIBRATION_MAP = {1: "AA001", 6: "AA006", 2: "AD002"}
SAMPLE_AREA = 5.0e-07
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"
FIDAS_INTEGRATION_MIN = 1.0
FIGURE_DPI = 300

WIND_DATA_PATTERN = re.compile(
    r'"?(2026-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"?,\s*'
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"([\d.]+),\s*"
    r"(\d+),\s*"
    r"(\d+)"
)

UCASS_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 6: "#2ca02c"}
TSI_COLOR = "#d62728"
FIDAS_COLOR = "#9467bd"


@dataclass(frozen=True)
class WindSpeedClass:
    file_tag: str
    label: str
    mode: Literal["range", "lt", "gt"] = "range"
    ws_min: float | None = None
    ws_max: float | None = None
    threshold: float | None = None


WIND_SPEED_CLASSES = (
    WindSpeedClass("0_2", "0–2 m/s", ws_min=0.0, ws_max=2.0),
    WindSpeedClass("2_4", "2–4 m/s", ws_min=2.0, ws_max=4.0),
    WindSpeedClass("4_6", "4–6 m/s", ws_min=4.0, ws_max=6.0),
    WindSpeedClass("6_8", "6–8 m/s", ws_min=6.0, ws_max=8.0),
    WindSpeedClass("8_10", "8–10 m/s", ws_min=8.0, ws_max=10.0),
    WindSpeedClass("10_12", "10–12 m/s", ws_min=10.0, ws_max=12.0),
    WindSpeedClass("lt_4", "< 4 m/s", mode="lt", threshold=4.0),
    WindSpeedClass("gt_4", "> 4 m/s", mode="gt", threshold=4.0),
)


# ---------------------------------------------------------------------------
# Wind-speed filter
# ---------------------------------------------------------------------------

def wind_speed_mask(ws: pd.Series, wind_class: WindSpeedClass) -> pd.Series:
    values = pd.to_numeric(ws, errors="coerce")
    if wind_class.mode == "lt":
        return values < wind_class.threshold
    if wind_class.mode == "gt":
        return values > wind_class.threshold
    return (values >= wind_class.ws_min) & (values < wind_class.ws_max)


# ---------------------------------------------------------------------------
# Shared loaders (mirrors overlap_period_dN_dlnDp_comparison_FIDAS_UCASS_TSI.py)
# ---------------------------------------------------------------------------

def load_bins_table(bins_file: Path) -> pd.DataFrame:
    if bins_file.suffix.lower() == ".xlsx":
        return pd.read_excel(bins_file, header=None)
    from numbers_parser import Document
    doc = Document(bins_file)
    table = doc.sheets[0].tables[0]
    return pd.DataFrame([
        [table.cell(r, c).value for c in range(table.num_cols)]
        for r in range(table.num_rows)
    ])


def load_calibrations() -> dict:
    bins_file = BINS_NUMBERS if BINS_NUMBERS.is_file() else BINS_XLSX
    if not bins_file.is_file():
        raise FileNotFoundError(f"Calibration file not found: {BINS_NUMBERS} or {BINS_XLSX}")
    raw = load_bins_table(bins_file)
    mapping = {"AA001": (0, 1), "AA006": (2, 3), "AD002": (4, 5), "AD005": (6, 7)}
    calibrations = {}
    for name, (col_low, col_up) in mapping.items():
        lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
        upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)
        calibrations[name] = {"centre": np.sqrt(lower * upper)}
    return calibrations


def load_ucass_csv(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=sep, low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def extract_ucass_counts(df: pd.DataFrame, uid: int, id_col: str, bin_suffix: str) -> pd.DataFrame:
    count_cols = [f"b{i}" for i in range(1, 16)]
    raw_cols = [f"b{i}{bin_suffix}" for i in range(1, 16)]
    sub = df.loc[df[id_col] == uid, ["Timestamp"] + raw_cols].copy()
    sub = sub.rename(columns=dict(zip(raw_cols, count_cols)))
    if sub["Timestamp"].duplicated().any():
        sub = sub.groupby("Timestamp", as_index=False)[count_cols].sum()
    return sub.sort_values("Timestamp").reset_index(drop=True)


def load_wind_rtf(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        m = WIND_DATA_PATTERN.search(line.strip().rstrip("\\"))
        if m:
            rows.append(m.groups())
    df = pd.DataFrame(rows, columns=[
        "TIMESTAMP", "RECORD", "AirTC_Avg", "RH", "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg",
    ])
    df["Timestamp"] = pd.to_datetime(df["TIMESTAMP"])
    df["WS_ms_Avg"] = pd.to_numeric(df["WS_ms_Avg"], errors="coerce")
    return df.dropna(subset=["Timestamp", "WS_ms_Avg"]).sort_values("Timestamp")


def build_ucass_master() -> pd.DataFrame:
    df_wind = load_wind_rtf(WIND_RTF)
    raw_cache: dict = {}
    master = None
    for uid in UCASS_IDS:
        cfg = UCASS_SOURCES[uid]
        key = (cfg["csv"], cfg["sep"])
        if key not in raw_cache:
            raw_cache[key] = load_ucass_csv(cfg["csv"], cfg["sep"])
        counts = extract_ucass_counts(
            raw_cache[key], uid, cfg["id_col"], cfg["bin_suffix"]
        )
        renamed = counts.rename(columns={f"b{i}": f"b{i}_id{uid}" for i in range(1, 16)})
        master = renamed if master is None else pd.merge(
            master, renamed, on="Timestamp", how="inner", validate="one_to_one"
        )
    master = pd.merge(
        master,
        df_wind[["Timestamp", "WS_ms_Avg"]],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )
    master["sample_vol_cm3"] = SAMPLE_AREA * master["WS_ms_Avg"] * 1e6
    return master


def attach_nearest_wind(df: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    left = df.sort_values("Timestamp").copy()
    right = wind[["Timestamp", "WS_ms_Avg"]].sort_values("Timestamp")
    merged = pd.merge_asof(
        left,
        right,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30s"),
    )
    return merged


def load_fidas() -> pd.DataFrame:
    return load_fidas_excel(FIDAS_XLSX)


def ucass_dln_dp(centres: np.ndarray) -> np.ndarray:
    widths = np.log(centres[1:]) - np.log(centres[:-1])
    return np.append(widths, widths[-1])


# ---------------------------------------------------------------------------
# Per-instrument PSD for a wind class
# ---------------------------------------------------------------------------

@dataclass
class InstrumentPSD:
    label: str
    diameters: np.ndarray
    dN_dlnDp: np.ndarray
    n_samples: int
    color: str
    markersize: float
    zorder: int


def compute_ucass_psd(
    master: pd.DataFrame,
    uid: int,
    centres: np.ndarray,
    mask: pd.Series,
) -> InstrumentPSD:
    subset = master.loc[mask].copy()
    n_samples = len(subset)
    if n_samples == 0:
        return InstrumentPSD(
            f"UCASS {uid}", centres, np.full_like(centres, np.nan, dtype=float),
            0, UCASS_COLORS[uid], 5.0, 3,
        )

    sample_vol = subset["sample_vol_cm3"].to_numpy(dtype=float)
    counts = subset[[f"b{i}_id{uid}" for i in range(1, 16)]].to_numpy(dtype=float)
    conc = counts.sum(axis=0) / sample_vol.sum()
    dln = ucass_dln_dp(centres)
    return InstrumentPSD(
        f"UCASS {uid}", centres, conc / dln, n_samples, UCASS_COLORS[uid], 5.0, 3,
    )


def compute_fidas_psd(
    fidas: pd.DataFrame,
    psd_cols: list[str],
    centres: np.ndarray,
    dln: np.ndarray,
    mask: pd.Series,
) -> InstrumentPSD:
    subset = fidas.loc[mask].copy()
    psd = subset[psd_cols].apply(pd.to_numeric, errors="coerce")
    valid = psd.notna().all(axis=1)
    subset = subset.loc[valid]
    psd = psd.loc[valid]
    n_samples = len(subset)

    if n_samples == 0:
        return InstrumentPSD(
            "FIDAS 200", centres, np.full_like(centres, np.nan, dtype=float),
            0, FIDAS_COLOR, 3.0, 5,
        )

    flow_lpm = pd.to_numeric(subset["Flowrate_Lpm"], errors="coerce").to_numpy(float)
    vol_L = flow_lpm * FIDAS_INTEGRATION_MIN
    dn = psd.to_numpy(dtype=float)
    conc = (dn * vol_L[:, None]).sum(axis=0) / vol_L.sum()
    return InstrumentPSD(
        "FIDAS 200", centres, conc / dln, n_samples, FIDAS_COLOR, 3.0, 5,
    )


def compute_tsi_psd(
    spectra: pd.DataFrame,
    bins: pd.DataFrame,
    centres: np.ndarray,
    dln: np.ndarray,
    mask: pd.Series,
) -> InstrumentPSD:
    subset = spectra.loc[mask].copy()
    conc_cols = [f"conc_Bin_{int(b)}" for b in bins["Bin"]]
    conc_data = subset[conc_cols].apply(pd.to_numeric, errors="coerce")
    valid = conc_data.notna().all(axis=1)
    subset = subset.loc[valid]
    conc_data = conc_data.loc[valid]
    n_samples = len(subset)

    if n_samples == 0:
        return InstrumentPSD(
            "TSI OPS 3330", centres, np.full_like(centres, np.nan, dtype=float),
            0, TSI_COLOR, 3.0, 5,
        )

    vol_cm3 = subset["sample_vol_cm3"].to_numpy(dtype=float)
    conc = (conc_data.to_numpy(dtype=float) * vol_cm3[:, None]).sum(axis=0) / vol_cm3.sum()
    return InstrumentPSD(
        "TSI OPS 3330", centres, conc / dln, n_samples, TSI_COLOR, 3.0, 5,
    )


# ---------------------------------------------------------------------------
# Plotting and export
# ---------------------------------------------------------------------------

def format_legend_label(psd: InstrumentPSD) -> str:
    return f"{psd.label} (n = {psd.n_samples:,})"


def plot_wind_class(
    spectra: list[InstrumentPSD],
    wind_class: WindSpeedClass,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for psd in spectra:
        y_plot = mask_nonpositive_for_log(psd.dN_dlnDp)
        if np.all(np.isnan(y_plot)):
            continue
        ax.loglog(
            psd.diameters,
            y_plot,
            "-o",
            markersize=psd.markersize,
            linewidth=1.2,
            color=psd.color,
            label=format_legend_label(psd),
            zorder=psd.zorder,
        )

    # Always show all instruments in legend, including n = 0
    handles = []
    labels = []
    for psd in spectra:
        line, = ax.plot([], [], "-o", color=psd.color, markersize=psd.markersize, linewidth=1.2)
        handles.append(line)
        labels.append(format_legend_label(psd))

    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN/dlnDp (# cm$^{-3}$)")
    ax.set_title(
        "FIDAS vs UCASS vs TSI — mean number size distribution (dN/dlnDp)\n"
        f"Wind speed: {wind_class.label}"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_csv(
    spectra: list[InstrumentPSD],
    wind_class: WindSpeedClass,
    output_csv: Path,
) -> None:
    frames = []
    for psd in spectra:
        frame = pd.DataFrame({
            "Wind_Speed_Class": wind_class.label,
            "Instrument": psd.label,
            "Diameter_um": psd.diameters,
            "dN_dlnDp": psd.dN_dlnDp,
            "N_Samples": psd.n_samples,
        })
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wind = load_wind_rtf(WIND_RTF)
    master = build_ucass_master()
    cal = load_calibrations()
    ucass_bins = {uid: cal[CALIBRATION_MAP[uid]]["centre"][1:] for uid in UCASS_IDS}

    fidas_raw = load_fidas()
    fidas = attach_nearest_wind(fidas_raw, wind)
    psd_cols = sorted_psd_columns(fidas.columns)
    fidas_centres = np.array([diameter_from_column(c) for c in psd_cols])
    fidas_lower, fidas_upper = geometric_bin_boundaries(fidas_centres)
    fidas_dln = np.log(fidas_upper / fidas_lower)

    tsi_spectra, _, tsi_bins = load_tsi_spectra()
    tsi = attach_nearest_wind(tsi_spectra, wind)
    tsi_centres = tsi_bins["Diameter_um"].to_numpy(dtype=float)
    tsi_dln = np.log(tsi_bins["Upper_um"] / tsi_bins["Lower_um"]).to_numpy(dtype=float)

    print("Mean dN/dlnDp by wind-speed class")
    print("=" * 60)

    for wind_class in WIND_SPEED_CLASSES:
        ucass_mask = wind_speed_mask(master["WS_ms_Avg"], wind_class)
        fidas_mask = wind_speed_mask(fidas["WS_ms_Avg"], wind_class)
        tsi_mask = wind_speed_mask(tsi["WS_ms_Avg"], wind_class)

        spectra = [
            compute_fidas_psd(fidas, psd_cols, fidas_centres, fidas_dln, fidas_mask),
        ]
        for uid in UCASS_IDS:
            spectra.append(
                compute_ucass_psd(master, uid, ucass_bins[uid], ucass_mask)
            )
        spectra.append(
            compute_tsi_psd(tsi, tsi_bins, tsi_centres, tsi_dln, tsi_mask)
        )

        png_path = OUTPUT_DIR / f"mean_dN_dlnDp_WS_{wind_class.file_tag}.png"
        csv_path = OUTPUT_DIR / f"mean_dN_dlnDp_WS_{wind_class.file_tag}.csv"

        plot_wind_class(spectra, wind_class, png_path)
        save_csv(spectra, wind_class, csv_path)

        counts = ", ".join(f"{psd.label} n={psd.n_samples:,}" for psd in spectra)
        print(f"{wind_class.label:<12}  {counts}")
        print(f"  -> {png_path.name}, {csv_path.name}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
