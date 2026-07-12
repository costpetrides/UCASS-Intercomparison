#!/usr/bin/env python3
"""
Compute and plot the mean particle size distribution (PSD) from a FIDAS 200
Excel export file.

Detects PSD columns automatically (e.g. dN0_1037, dN12_5432), extracts bin
diameters from column names, averages all valid spectra, and writes a CSV
plus a publication-quality figure.

Dependencies: numpy, pandas, matplotlib, openpyxl
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from fidas_utils import (
    build_psd_table,
    detect_psd_columns,
    filter_fidas_to_ucass_period,
    load_fidas_excel,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "fidas" / "FIDAS200.txt"
DEFAULT_SHEET = 0  # first sheet

OUTPUT_CSV = ROOT / "outputs" / "fidas" / "FIDAS_mean_PSD.csv"
OUTPUT_PNG = ROOT / "outputs" / "fidas" / "FIDAS_mean_PSD.png"

FIGURE_DPI = 300


def compute_mean_psd(
    df: pd.DataFrame,
    psd_table: pd.DataFrame,
) -> tuple[pd.Series, int]:
    """
    Drop rows with any NaN in PSD columns, then return the mean spectrum.

    Returns
    -------
    mean_dn : Series indexed by Diameter_um
    n_valid : number of spectra averaged
    """
    psd_columns = psd_table["column"].tolist()
    psd_data = df[psd_columns].apply(pd.to_numeric, errors="coerce")
    valid = psd_data.dropna(how="any")

    if valid.empty:
        raise ValueError(
            "No valid spectra remain after removing rows with NaN in PSD columns."
        )

    # Mean along time axis; align output to sorted diameters
    mean_by_column = valid.mean(axis=0)
    mean_dn = pd.Series(
        mean_by_column.values,
        index=psd_table["Diameter_um"].values,
        name="Mean_dN",
    )
    return mean_dn, len(valid)


# ---------------------------------------------------------------------------
# Summary and plotting
# ---------------------------------------------------------------------------

def print_summary(
    psd_table: pd.DataFrame,
    n_valid: int,
) -> None:
    """Print diagnostic summary before plotting."""
    diameters = psd_table["Diameter_um"]
    columns = psd_table["column"].tolist()

    print("FIDAS mean PSD — summary")
    print("=" * 40)
    print(f"Total number of PSD bins     : {len(psd_table)}")
    print(f"Minimum particle diameter    : {diameters.min():.4f} µm")
    print(f"Maximum particle diameter    : {diameters.max():.4f} µm")
    print(f"Number of valid spectra      : {n_valid}")
    print("Detected PSD columns:")
    for col in columns:
        print(f"  {col}")
    print()


def plot_mean_psd(
    mean_psd: pd.Series,
    output_path: Path,
) -> None:
    """
    Create a publication-quality mean PSD figure.

    Uses dN values exactly as stored in the dataset (not normalised).
    """
    diameters = mean_psd.index.to_numpy(dtype=float)
    values = mean_psd.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        diameters,
        values,
        marker="o",
        markersize=4,
        linewidth=1.2,
        color="steelblue",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_mean_psd_csv(mean_psd: pd.Series, output_path: Path) -> None:
    """Write mean PSD to CSV with columns Diameter_um and Mean_dN."""
    out = pd.DataFrame(
        {
            "Diameter_um": mean_psd.index,
            "Mean_dN": mean_psd.values,
        }
    )
    out.to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path = DEFAULT_INPUT) -> None:
    """Load FIDAS data, compute mean PSD, save outputs."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_fidas_excel(input_path)
    df, period_start, period_end = filter_fidas_to_ucass_period(df)
    print(
        f"FIDAS restricted to UCASS period: "
        f"{period_start} -> {period_end} UTC ({len(df)} spectra)"
    )

    psd_columns = detect_psd_columns(df.columns)
    if not psd_columns:
        raise ValueError(
            "No PSD columns found. Expected dN0_1037-style or numeric-centre columns."
        )

    psd_table = build_psd_table(psd_columns)
    mean_psd, n_valid = compute_mean_psd(df, psd_table)

    print_summary(psd_table, n_valid)

    save_mean_psd_csv(mean_psd, OUTPUT_CSV)
    plot_mean_psd(mean_psd, OUTPUT_PNG)

    print(f"Saved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    main(input_file)
