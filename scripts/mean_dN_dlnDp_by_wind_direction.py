#!/usr/bin/env python3
"""
Mean dN/dlnDp PSDs stratified by wind-direction class for all reference instruments.

For each direction bin, volume-weighted mean spectra are computed for
FIDAS 200, UCASS 1/2/6, and TSI OPS 3330, then plotted with the same style
as combined_FIDAS_UCASS_TSI_dN_dlnDp.png. Sample counts appear in the legend.

Outputs (per class):
  outputs/combined/wind_direction/mean_dN_dlnDp_WD_<class>.png
  outputs/combined/wind_direction/mean_dN_dlnDp_WD_<class>.csv

Dependencies: numpy, pandas, matplotlib, openpyxl
Optional: numbers-parser (if UCASS_size_bins.numbers is used for calibrations)
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

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
OUTPUT_DIR = ROOT / "outputs" / "combined" / "wind_direction"

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"
FIDAS_XLSX = ROOT / "data" / "fidas" / "FIDAS200.txt"

UCASS_IDS = (1, 2, 6)
UCASS_SOURCES = {
    1: {"csv": UCASS27_CSV, "sep": ",", "id_col": "UCASS_ID", "bin_suffix": ""},
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
class WindDirectionClass:
    file_tag: str
    label: str
    dir_min: float
    dir_max: float
    wrap: bool = False


# Eight compass sectors; N wraps across 0°. Intervals are half-open [min, max).
WIND_DIRECTION_CLASSES = (
    WindDirectionClass("N", "N (337.5–360° and 0–22.5°)", 337.5, 22.5, wrap=True),
    WindDirectionClass("NE", "NE (22.5–67.5°)", 22.5, 67.5),
    WindDirectionClass("E", "E (67.5–112.5°)", 67.5, 112.5),
    WindDirectionClass("SE", "SE (112.5–157.5°)", 112.5, 157.5),
    WindDirectionClass("S", "S (157.5–202.5°)", 157.5, 202.5),
    WindDirectionClass("SW", "SW (202.5–247.5°)", 202.5, 247.5),
    WindDirectionClass("W", "W (247.5–292.5°)", 247.5, 292.5),
    WindDirectionClass("NW", "NW (292.5–337.5°)", 292.5, 337.5),
)


# ---------------------------------------------------------------------------
# Wind-direction filter
# ---------------------------------------------------------------------------

def wind_direction_mask(wind_dir: pd.Series, direction_class: WindDirectionClass) -> pd.Series:
    """Half-open compass sector in degrees; N sector wraps across 0°."""
    values = pd.to_numeric(wind_dir, errors="coerce")
    if direction_class.wrap:
        return (values >= direction_class.dir_min) | (values < direction_class.dir_max)
    return (values >= direction_class.dir_min) & (values < direction_class.dir_max)


# ---------------------------------------------------------------------------
# Shared loaders
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
    df["WindDir"] = pd.to_numeric(df["WindDir"], errors="coerce")
    return df.dropna(subset=["Timestamp", "WS_ms_Avg", "WindDir"]).sort_values("Timestamp")


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
        df_wind[["Timestamp", "WS_ms_Avg", "WindDir"]],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )
    master["sample_vol_cm3"] = SAMPLE_AREA * master["WS_ms_Avg"] * 1e6
    return master


def attach_nearest_meteorology(df: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    left = df.sort_values("Timestamp").copy()
    for col in ("WS_ms_Avg", "WindDir"):
        if col in left.columns:
            left = left.drop(columns=[col])
    right = wind[["Timestamp", "WS_ms_Avg", "WindDir"]].sort_values("Timestamp")
    return pd.merge_asof(
        left,
        right,
        on="Timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30s"),
    )


def load_fidas() -> pd.DataFrame:
    return load_fidas_excel(FIDAS_XLSX)


def ucass_dln_dp(centres: np.ndarray) -> np.ndarray:
    widths = np.log(centres[1:]) - np.log(centres[:-1])
    return np.append(widths, widths[-1])


# ---------------------------------------------------------------------------
# Per-instrument PSD for a direction class
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


def plot_direction_class(
    spectra: list[InstrumentPSD],
    direction_class: WindDirectionClass,
    output_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for psd in spectra:
        y_plot = mask_nonpositive_for_log(psd.dN_dlnDp)
        if not np.all(np.isnan(y_plot)):
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
        f"Wind direction: {direction_class.label}"
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(handles, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=FIGURE_DPI)
    plt.close(fig)


def save_csv(
    spectra: list[InstrumentPSD],
    direction_class: WindDirectionClass,
    output_csv: Path,
) -> None:
    frames = []
    for psd in spectra:
        frames.append(pd.DataFrame({
            "Wind_Direction_Class": direction_class.label,
            "Instrument": psd.label,
            "Diameter_um": psd.diameters,
            "dN_dlnDp": psd.dN_dlnDp,
            "N_Samples": psd.n_samples,
        }))
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
    fidas = attach_nearest_meteorology(fidas_raw, wind)
    psd_cols = sorted_psd_columns(fidas.columns)
    fidas_centres = np.array([diameter_from_column(c) for c in psd_cols])
    fidas_lower, fidas_upper = geometric_bin_boundaries(fidas_centres)
    fidas_dln = np.log(fidas_upper / fidas_lower)

    tsi_spectra, _, tsi_bins = load_tsi_spectra()
    tsi = attach_nearest_meteorology(tsi_spectra, wind)
    tsi_centres = tsi_bins["Diameter_um"].to_numpy(dtype=float)
    tsi_dln = np.log(tsi_bins["Upper_um"] / tsi_bins["Lower_um"]).to_numpy(dtype=float)

    print("Mean dN/dlnDp by wind-direction class")
    print("=" * 60)

    for direction_class in WIND_DIRECTION_CLASSES:
        ucass_mask = wind_direction_mask(master["WindDir"], direction_class)
        fidas_mask = wind_direction_mask(fidas["WindDir"], direction_class)
        tsi_mask = wind_direction_mask(tsi["WindDir"], direction_class)

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

        png_path = OUTPUT_DIR / f"mean_dN_dlnDp_WD_{direction_class.file_tag}.png"
        csv_path = OUTPUT_DIR / f"mean_dN_dlnDp_WD_{direction_class.file_tag}.csv"

        plot_direction_class(spectra, direction_class, png_path)
        save_csv(spectra, direction_class, csv_path)

        counts = ", ".join(f"{psd.label} n={psd.n_samples:,}" for psd in spectra)
        print(f"{direction_class.label:<22}  {counts}")
        print(f"  -> {png_path.name}, {csv_path.name}")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
