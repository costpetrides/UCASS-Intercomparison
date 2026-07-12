#!/usr/bin/env python3
"""
Overlay FIDAS, UCASS, and TSI mean dN/dlnDp PSDs on a single figure.

Reads pre-computed outputs only (does not modify existing analyses):

  outputs/fidas/FIDAS_mean_dN_dlnDp.csv
  outputs/ucass/UCASS{1,2,6}_integrated_psd.csv
  outputs/tsi/TSI_mean_dN_dlnDp.csv

Dependencies: numpy, pandas, matplotlib
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

from plot_utils import mask_nonpositive_for_log

ROOT = Path(__file__).resolve().parent.parent
UCASS_OUTPUT_DIR = ROOT / "outputs" / "ucass"

FIDAS_CSV = ROOT / "outputs" / "fidas" / "FIDAS_mean_dN_dlnDp.csv"
TSI_CSV = ROOT / "outputs" / "tsi" / "TSI_mean_dN_dlnDp.csv"
UCASS_IDS = (1, 2, 6)
OUTPUT_PNG = ROOT / "outputs" / "combined" / "combined_FIDAS_UCASS_TSI_dN_dlnDp.png"

FIGURE_DPI = 300

UCASS_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    6: "#2ca02c",
}
TSI_COLOR = "#d62728"


def load_fidas(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"FIDAS dN/dlnDp CSV not found: {path}\n"
            "Run fidas_mean_dN_dlnDp.py first."
        )
    df = pd.read_csv(path)
    required = {"Diameter_um", "dN_dlnDp"}
    if not required.issubset(df.columns):
        raise ValueError(f"FIDAS CSV must contain columns {required}")
    return df.sort_values("Diameter_um").reset_index(drop=True)


def load_tsi(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"TSI dN/dlnDp CSV not found: {path}\n"
            "Run tsi_mean_dN_dlnDp.py first."
        )
    df = pd.read_csv(path)
    required = {"Diameter_um", "dN_dlnDp"}
    if not required.issubset(df.columns):
        raise ValueError(f"TSI CSV must contain columns {required}")
    return df.sort_values("Diameter_um").reset_index(drop=True)


def load_ucass(uid: int) -> pd.DataFrame:
    path = UCASS_OUTPUT_DIR / f"UCASS{uid}_integrated_psd.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"UCASS PSD CSV not found: {path}\n"
            "Run Intercomparison.py first."
        )
    df = pd.read_csv(path)
    if "Dp_um" not in df.columns or "dN_dlnDp_cm3" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}")
    return df.rename(columns={
        "Dp_um": "Diameter_um",
        "dN_dlnDp_cm3": "dN_dlnDp",
    }).sort_values("Diameter_um").reset_index(drop=True)


def plot_combined(
    fidas: pd.DataFrame,
    ucass: dict[int, pd.DataFrame],
    tsi: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        fidas["Diameter_um"],
        mask_nonpositive_for_log(fidas["dN_dlnDp"].to_numpy()),
        "-o",
        markersize=3,
        linewidth=1.2,
        color="#9467bd",
        label="FIDAS 200",
        zorder=5,
    )

    for uid in UCASS_IDS:
        df = ucass[uid]
        ax.loglog(
            df["Diameter_um"],
            mask_nonpositive_for_log(df["dN_dlnDp"].to_numpy()),
            "-o",
            markersize=5,
            linewidth=1.2,
            color=UCASS_COLORS[uid],
            label=f"UCASS {uid}",
        )

    ax.loglog(
        tsi["Diameter_um"],
        mask_nonpositive_for_log(tsi["dN_dlnDp"].to_numpy()),
        "-o",
        markersize=3,
        linewidth=1.2,
        color=TSI_COLOR,
        label="TSI OPS 3330",
        zorder=5,
    )

    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN/dlnDp (# cm$^{-3}$)")
    ax.set_title("FIDAS vs UCASS vs TSI — mean number size distribution (dN/dlnDp)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> None:
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fidas = load_fidas(FIDAS_CSV)
    tsi = load_tsi(TSI_CSV)
    ucass = {uid: load_ucass(uid) for uid in UCASS_IDS}

    print("Combined FIDAS / UCASS / TSI dN/dlnDp overlay")
    print("=" * 40)
    print(f"FIDAS bins : {len(fidas)}")
    print(f"TSI bins   : {len(tsi)}")
    for uid in UCASS_IDS:
        df = ucass[uid]
        print(
            f"UCASS {uid}  : {len(df)} bins, "
            f"{df['Diameter_um'].min():.3f}–{df['Diameter_um'].max():.3f} µm"
        )
    print(f"FIDAS range: {fidas['Diameter_um'].min():.4f}–{fidas['Diameter_um'].max():.4f} µm")
    print(f"TSI range  : {tsi['Diameter_um'].min():.4f}–{tsi['Diameter_um'].max():.4f} µm")

    plot_combined(fidas, ucass, tsi, OUTPUT_PNG)
    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
