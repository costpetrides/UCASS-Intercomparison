#!/usr/bin/env python3
"""
Mean total particle concentration over the common overlapping period.

For each instrument, restrict to the identical overlap window, compute total
particle concentration (#/cm³) at every timestamp over the TSI-matched
sized range (0.3–10 µm), then take the arithmetic mean over available
measurements.

Outputs:
  outputs/overlap/mean_total_particle_concentration.png
  outputs/overlap/mean_total_particle_concentration.csv

Dependencies: numpy, pandas, matplotlib, openpyxl
Optional: numbers-parser (if UCASS_size_bins.numbers is used for calibrations)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tsi_mean_psd import load_tsi_spectra
from fidas_utils import (
    diameter_from_column,
    filter_to_reference_timestamps,
    is_psd_column,
    load_fidas_excel,
    reference_period_bounds,
    ucass_reference_timestamps,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"
FIDAS_XLSX = ROOT / "data" / "fidas" / "FIDAS200.txt"

# TSI OPS 3330 sized-channel span (bins 1–16); used to match UCASS/FIDAS N_tot
TSI_SIZE_LO_UM = 0.3
TSI_SIZE_HI_UM = 10.0

CALIBRATION_MAP = {1: "AA001", 6: "AA006", 2: "AD002"}
UCASS_BIN_COLUMNS = {
    "AA001": (0, 1),
    "AA006": (2, 3),
    "AD002": (4, 5),
}

OUTPUT_CSV = ROOT / "outputs" / "overlap" / "mean_total_particle_concentration.csv"
OUTPUT_PNG = ROOT / "outputs" / "overlap" / "mean_total_particle_concentration.png"

UCASS_IDS = (1, 2, 6)
UCASS_SOURCES = {
    1: {"csv": UCASS27_CSV, "sep": ";", "id_col": "UCASS_ID", "bin_suffix": ""},
    2: {"csv": UCASS362_CSV, "sep": ";", "id_col": "UCASS_ID.1", "bin_suffix": ".1"},
    6: {"csv": UCASS362_CSV, "sep": ";", "id_col": "UCASS_ID", "bin_suffix": ""},
}
SAMPLE_AREA = 5.0e-07
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"
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

INSTRUMENT_ORDER = [
    ("UCASS 1", "UCASS", 1),
    ("UCASS 2", "UCASS", 2),
    ("UCASS 6", "UCASS", 6),
    ("FIDAS 200", "FIDAS", None),
    ("TSI OPS 3330", "TSI", None),
]

BAR_COLORS = {
    "UCASS 1": "#1f77b4",
    "UCASS 2": "#ff7f0e",
    "UCASS 6": "#2ca02c",
    "FIDAS 200": "#9467bd",
    "TSI OPS 3330": "#d62728",
}


# ---------------------------------------------------------------------------
# Reference timestamps (UCASS 1/2/6 + wind exact overlap)
# ---------------------------------------------------------------------------

def select_reference_period(
    master: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.Timestamp, pd.Timestamp]:
    reference = ucass_reference_timestamps(master)
    period_start, period_end = reference_period_bounds(reference)
    return reference, period_start, period_end


# ---------------------------------------------------------------------------
# UCASS
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


def load_ucass_csv(path: Path, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=4, sep=sep, low_memory=False)
    if "GPS_Date" not in df.columns:
        alt_sep = "," if sep == ";" else ";"
        df = pd.read_csv(path, skiprows=4, sep=alt_sep, low_memory=False)
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
    return df.dropna(subset=["Timestamp"])


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


def load_ucass_bin_boundaries(uid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bin edges (µm) for UCASS uid, matching Intercomparison.py (skip row 0)."""
    bins_file = BINS_XLSX if BINS_XLSX.is_file() else BINS_NUMBERS
    raw = load_bins_table(bins_file)
    cal_name = CALIBRATION_MAP[uid]
    col_low, col_up = UCASS_BIN_COLUMNS[cal_name]
    lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)[1:]
    upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)[1:]
    return lower, upper


def ucass_bins_overlapping_tsi_range(uid: int) -> list[int]:
    """Bin numbers (1–15) whose boundaries overlap the TSI sized range."""
    lower, upper = load_ucass_bin_boundaries(uid)
    overlap = (lower < TSI_SIZE_HI_UM) & (upper > TSI_SIZE_LO_UM)
    return [i + 1 for i, ok in enumerate(overlap) if ok]


def fidas_psd_columns_in_tsi_range(columns: pd.Index) -> list[str]:
    psd_cols = [str(c) for c in columns if is_psd_column(str(c))]
    return [
        col for col in psd_cols
        if TSI_SIZE_LO_UM <= diameter_from_column(col) <= TSI_SIZE_HI_UM
    ]


def ucass_mean_total_concentration(
    master: pd.DataFrame,
    uid: int,
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[float, int, str]:
    period = filter_to_reference_timestamps(master, reference_timestamps).copy()
    bin_nums = ucass_bins_overlapping_tsi_range(uid)
    count_cols = [f"b{i}_id{uid}" for i in bin_nums]
    counts = period[count_cols].to_numpy(dtype=float)
    sample_vol = period["sample_vol_cm3"].to_numpy(dtype=float)
    valid = sample_vol > 0
    if not valid.any():
        size_label = f"{TSI_SIZE_LO_UM}–{TSI_SIZE_HI_UM} µm ({len(bin_nums)} UCASS bins)"
        return float("nan"), len(period), size_label
    total_vol = sample_vol[valid].sum()
    total_counts = counts[valid].sum()
    size_label = f"{TSI_SIZE_LO_UM}–{TSI_SIZE_HI_UM} µm ({len(bin_nums)} UCASS bins)"
    return float(total_counts / total_vol), int(valid.sum()), size_label


# ---------------------------------------------------------------------------
# FIDAS
# ---------------------------------------------------------------------------

def load_fidas() -> pd.DataFrame:
    return load_fidas_excel(FIDAS_XLSX)


def fidas_mean_total_concentration(
    df: pd.DataFrame,
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[float, int, str]:
    period = filter_to_reference_timestamps(df, reference_timestamps).copy()
    psd_cols = fidas_psd_columns_in_tsi_range(period.columns)
    psd = period[psd_cols].apply(pd.to_numeric, errors="coerce")
    valid = psd.notna().all(axis=1)
    psd = psd.loc[valid]

    # Sum per-bin #/cm³ over TSI-matched diameters (not full-span Cn)
    n_tot = psd.sum(axis=1)
    size_label = f"{TSI_SIZE_LO_UM}–{TSI_SIZE_HI_UM} µm ({len(psd_cols)} FIDAS bins)"
    return float(n_tot.mean()), len(psd), size_label


# ---------------------------------------------------------------------------
# TSI
# ---------------------------------------------------------------------------

def tsi_mean_total_concentration(
    spectra: pd.DataFrame,
    bins: pd.DataFrame,
    reference_timestamps: pd.DatetimeIndex,
) -> tuple[float, int, str]:
    period = filter_to_reference_timestamps(spectra, reference_timestamps).copy()
    conc_cols = [f"conc_Bin_{int(b)}" for b in bins["Bin"]]
    conc_data = period[conc_cols].apply(pd.to_numeric, errors="coerce")
    valid = conc_data.notna().all(axis=1)
    conc_data = conc_data.loc[valid]
    n_tot = conc_data.sum(axis=1)
    lo = float(bins["Lower_um"].min())
    hi = float(bins["Upper_um"].max())
    size_label = f"{lo}–{hi} µm ({len(bins)} TSI bins)"
    return float(n_tot.mean()), len(conc_data), size_label


# ---------------------------------------------------------------------------
# Overlap window and outputs
# ---------------------------------------------------------------------------

def plot_summary(summary: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [BAR_COLORS[label] for label in summary["Instrument"]]

    ax.bar(summary["Instrument"], summary["Mean_Total_Concentration_cm3"], color=colors)
    ax.set_ylabel("Mean Total Particle Concentration (# cm$^{-3}$)")
    ax.set_title(
        "Mean Total Particle Concentration (Overlapping Period)\n"
        f"{period_start} – {period_end} UTC  |  sized range {TSI_SIZE_LO_UM}–{TSI_SIZE_HI_UM} µm"
    )
    ax.grid(True, axis="y")

    for bar, value in zip(ax.patches, summary["Mean_Total_Concentration_cm3"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    master = build_ucass_master()
    fidas_df = load_fidas()
    tsi_spectra, _, tsi_bins = load_tsi_spectra()

    reference, period_start, period_end = select_reference_period(master)

    rows = []
    for label, family, uid in INSTRUMENT_ORDER:
        if family == "UCASS":
            mean_conc, n_samples, size_range = ucass_mean_total_concentration(
                master, uid, reference,
            )
        elif family == "FIDAS":
            mean_conc, n_samples, size_range = fidas_mean_total_concentration(
                fidas_df, reference,
            )
        else:
            mean_conc, n_samples, size_range = tsi_mean_total_concentration(
                tsi_spectra, tsi_bins, reference,
            )

        rows.append({
            "Instrument": label,
            "Mean_Total_Concentration_cm3": mean_conc,
            "Units": "#/cm³",
            "Size_Range": size_range,
            "N_Samples": n_samples,
            "Period_Start_UTC": period_start,
            "Period_End_UTC": period_end,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_CSV, index=False)
    plot_summary(summary, period_start, period_end)

    print("Mean total particle concentration (UCASS-matched timestamps)")
    print("=" * 56)
    print(f"UCASS reference timestamps: {len(reference)}")
    print(f"Period bounds: {period_start} -> {period_end} UTC")
    print(summary.to_string(index=False))
    print(f"\nSaved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
