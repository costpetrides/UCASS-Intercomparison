#!/usr/bin/env python3
"""
Compute UCASS mean per-bin number concentration (#/cm³) for direct comparison
with FIDAS dN* channel concentrations.

Replicates the UCASS + wind processing pipeline from Intercomparison.py
(overlap period, sample volume, calibrations) but does NOT divide by ΔlnDp.
Produces additional outputs only; does not modify existing analysis or figures.

Dependencies: numpy, pandas, matplotlib, numbers-parser (for .numbers calibrations)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants (aligned with Intercomparison.py)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"

OUTPUT_CSV = ROOT / "outputs" / "ucass_mean" / "UCASS_mean_per_bin_PSD.csv"
OUTPUT_PNG = ROOT / "outputs" / "ucass_mean" / "UCASS_mean_per_bin_PSD.png"

UCASS_IDS = (1, 2, 6)

UCASS_SOURCES = {
    1: {
        "csv": UCASS27_CSV,
        "sep": ",",
        "id_col": "UCASS_ID",
        "bin_suffix": "",
    },
    2: {
        "csv": UCASS362_CSV,
        "sep": ",",
        "id_col": "UCASS_ID.1",
        "bin_suffix": ".1",
    },
    6: {
        "csv": UCASS362_CSV,
        "sep": ",",
        "id_col": "UCASS_ID",
        "bin_suffix": "",
    },
}

CALIBRATION_MAP = {
    1: "AA001",
    6: "AA006",
    2: "AD002",
}

SAMPLE_AREA = 5.0e-07          # m²
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


# ---------------------------------------------------------------------------
# Pipeline helpers (mirrored from Intercomparison.py)
# ---------------------------------------------------------------------------

def load_bins_table(bins_file: Path) -> pd.DataFrame:
    """Load calibration table from Apple Numbers or Excel (same layout)."""
    if bins_file.suffix.lower() == ".xlsx":
        return pd.read_excel(bins_file, header=None)

    from numbers_parser import Document

    doc = Document(bins_file)
    table = doc.sheets[0].tables[0]
    rows = [
        [table.cell(row_idx, col_idx).value for col_idx in range(table.num_cols)]
        for row_idx in range(table.num_rows)
    ]
    return pd.DataFrame(rows)


def load_calibrations() -> dict:
    if BINS_NUMBERS.is_file():
        bins_file = BINS_NUMBERS
    elif BINS_XLSX.is_file():
        bins_file = BINS_XLSX
    else:
        raise FileNotFoundError(
            f"Calibration file not found: {BINS_NUMBERS} or {BINS_XLSX}"
        )

    raw = load_bins_table(bins_file)
    mapping = {
        "AA001": (0, 1),
        "AA006": (2, 3),
        "AD002": (4, 5),
        "AD005": (6, 7),
    }
    calibrations = {}
    for name, (col_low, col_up) in mapping.items():
        lower = pd.to_numeric(raw.iloc[2:, col_low], errors="coerce").dropna().to_numpy(float)
        upper = pd.to_numeric(raw.iloc[2:, col_up], errors="coerce").dropna().to_numpy(float)
        centre = np.sqrt(lower * upper)
        calibrations[name] = {"lower": lower, "upper": upper, "centre": centre}
    return calibrations


def load_ucass_csv(ucass_file: Path, sep: str = ",") -> pd.DataFrame:
    df = pd.read_csv(ucass_file, skiprows=4, sep=sep, low_memory=False)
    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " + df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce",
    )
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def extract_ucass_counts(
    df: pd.DataFrame,
    ucass_id: int,
    id_col: str = "UCASS_ID",
    bin_suffix: str = "",
) -> pd.DataFrame:
    raw_count_cols = [f"b{i}{bin_suffix}" for i in range(1, 16)]
    count_cols = [f"b{i}" for i in range(1, 16)]
    subset = df.loc[df[id_col] == ucass_id, ["Timestamp"] + raw_count_cols].copy()
    subset = subset.rename(columns=dict(zip(raw_count_cols, count_cols)))

    if subset["Timestamp"].duplicated().any():
        subset = subset.groupby("Timestamp", as_index=False)[count_cols].sum()

    return subset.sort_values("Timestamp").reset_index(drop=True)


def load_wind_rtf(wind_file: Path) -> pd.DataFrame:
    rtf_text = wind_file.read_text(encoding="utf-8", errors="replace")
    rows = []
    for raw_line in rtf_text.split("\n"):
        match = WIND_DATA_PATTERN.search(raw_line.strip().rstrip("\\"))
        if match:
            rows.append(match.groups())
    if not rows:
        raise ValueError(f"No wind data rows found in {wind_file}")

    df = pd.DataFrame(
        rows,
        columns=[
            "TIMESTAMP", "RECORD", "AirTC_Avg", "RH",
            "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg",
        ],
    )
    df["Timestamp"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    for col in ["RECORD", "AirTC_Avg", "RH", "WindDir", "WS_ms_Avg", "WSDiag", "BP_mbar_Avg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)


def merge_multi_ucass_wind(
    ucass_frames: list[tuple[int, pd.DataFrame]],
    df_wind: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join all UCASS streams, then inner-join wind at 1 Hz."""
    master = None
    for uid, df in ucass_frames:
        renamed = df.rename(columns={f"b{i}": f"b{i}_id{uid}" for i in range(1, 16)})
        master = renamed if master is None else pd.merge(
            master, renamed, on="Timestamp", how="inner", validate="one_to_one"
        )

    wind_cols = ["Timestamp", "WS_ms_Avg", "WindDir", "AirTC_Avg", "RH", "BP_mbar_Avg"]
    wind_timestamps = set(df_wind["Timestamp"])
    if len(set(master["Timestamp"]) & wind_timestamps) != len(master):
        raise ValueError("Wind timestamp mismatch after UCASS merge.")

    return pd.merge(
        master, df_wind[wind_cols], on="Timestamp", how="inner", validate="one_to_one"
    )


def build_master_table() -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Load data and return merged overlap table plus bin centres per UCASS."""
    cal = load_calibrations()
    df_wind = load_wind_rtf(WIND_RTF)

    raw_cache: dict = {}
    ucass_frames = []
    for uid in UCASS_IDS:
        cfg = UCASS_SOURCES[uid]
        key = (cfg["csv"], cfg["sep"])
        if key not in raw_cache:
            raw_cache[key] = load_ucass_csv(cfg["csv"], sep=cfg["sep"])
        counts = extract_ucass_counts(
            raw_cache[key], uid, id_col=cfg["id_col"], bin_suffix=cfg["bin_suffix"]
        )
        ucass_frames.append((uid, counts))

    master = merge_multi_ucass_wind(ucass_frames, df_wind)
    master["sample_vol_cm3"] = SAMPLE_AREA * master["WS_ms_Avg"] * 1e6

    ucass_bins = {
        uid: cal[CALIBRATION_MAP[uid]]["centre"][1:]
        for uid in UCASS_IDS
    }
    return master, ucass_bins


def compute_mean_per_bin_concentrations(
    master: pd.DataFrame,
    ucass_bins: dict[int, np.ndarray],
) -> dict[int, pd.DataFrame]:
    """
    Per-second per-bin concentration = counts / sample_vol_cm3.
    Mean over the overlap period = arithmetic mean across timestamps.
    """
    results = {}
    sample_vol = master["sample_vol_cm3"].to_numpy(dtype=float)

    for uid in UCASS_IDS:
        count_cols = [f"b{i}_id{uid}" for i in range(1, 16)]
        counts = master[count_cols].to_numpy(dtype=float)
        per_bin_conc = counts / sample_vol[:, None]
        mean_per_bin = per_bin_conc.mean(axis=0)

        results[uid] = pd.DataFrame({
            "UCASS_ID": uid,
            "Diameter_um": ucass_bins[uid],
            "Mean_per_bin_conc_cm3": mean_per_bin,
        })

    return results


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_csv(per_ucass: dict[int, pd.DataFrame], path: Path) -> None:
    combined = pd.concat(per_ucass.values(), ignore_index=True)
    combined.to_csv(path, index=False)


def plot_mean_per_bin_psd(per_ucass: dict[int, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for uid in UCASS_IDS:
        df = per_ucass[uid]
        ax.plot(
            df["Diameter_um"],
            df["Mean_per_bin_conc_cm3"],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"UCASS {uid}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("Number concentration (#/cm³ per bin)")
    ax.set_title("UCASS mean per-bin number concentration")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)


def print_summary(master: pd.DataFrame, per_ucass: dict[int, pd.DataFrame]) -> None:
    print("UCASS mean per-bin PSD — summary")
    print("=" * 50)
    print(f"Analysis period : {master['Timestamp'].min()} -> {master['Timestamp'].max()}")
    print(f"Timestamps      : {len(master)}")
    print(f"Mean sample vol : {master['sample_vol_cm3'].mean():.4f} cm³/s")
    print()

    for uid in UCASS_IDS:
        df = per_ucass[uid]
        total_mean = df["Mean_per_bin_conc_cm3"].sum()
        print(f"UCASS {uid} ({CALIBRATION_MAP[uid]}):")
        print(f"  Bins            : {len(df)}")
        print(f"  Diameter range  : {df['Diameter_um'].min():.4f} – {df['Diameter_um'].max():.4f} µm")
        print(f"  Sum of bin means: {total_mean:.4f} #/cm³")
        print()


def print_fidas_comparability_note() -> None:
    print("Comparability with FIDAS dN-per-bin PSD")
    print("=" * 50)
    print(
        "Same quantity class: both are per-bin number concentrations (#/cm³)\n"
        "that sum to total number concentration without ΔlnDp or ΔlogDp.\n"
        "\n"
        "Direct overlay comparison is limited because:\n"
        "  • Different bin grids (UCASS: 15 bins; FIDAS: 78 bins).\n"
        "  • Different diameter ranges and calibrations (UCASS units use\n"
        "    AA001 / AD002 / AA006; FIDAS ~0.10–26 µm).\n"
        "  • Different sampling geometry (open-path UCASS vs FIDAS inlet).\n"
        "  • Possibly different time windows unless periods are aligned.\n"
        "\n"
        "Valid comparisons: total number concentration (sum of bin means),\n"
        "qualitative shape over overlapping diameters, and integrated counts\n"
        "in diameter ranges common to both instruments."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master, ucass_bins = build_master_table()
    per_ucass = compute_mean_per_bin_concentrations(master, ucass_bins)

    print_summary(master, per_ucass)

    save_csv(per_ucass, OUTPUT_CSV)
    plot_mean_per_bin_psd(per_ucass, OUTPUT_PNG)

    print(f"Saved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")
    print()
    print_fidas_comparability_note()


if __name__ == "__main__":
    main()
