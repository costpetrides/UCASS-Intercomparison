#!/usr/bin/env python3
"""
Overlay FIDAS, UCASS, and TSI mean dN/dlnDp PSDs on a single figure.

All instruments are averaged only at the exact UCASS reference timestamps
(UCASS 1/2/6 + wind inner overlap), using the same volume-weighted methods as
overlap_period_dN_dlnDp_comparison_FIDAS_UCASS_TSI.py.

Dependencies: numpy, pandas, matplotlib, openpyxl
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_utils import mask_nonpositive_for_log
from tsi_mean_psd import load_tsi_spectra

from overlap_period_dN_dlnDp_comparison_FIDAS_UCASS_TSI import (
    CALIBRATION_MAP,
    UCASS_COLORS,
    UCASS_IDS,
    TSI_COLOR,
    build_ucass_master,
    compute_fidas_overlap_psd,
    compute_tsi_overlap_psd,
    compute_ucass_overlap_psd,
    load_calibrations,
    load_fidas,
    select_reference_period,
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PNG = ROOT / "outputs" / "combined" / "combined_FIDAS_UCASS_TSI_dN_dlnDp.png"
FIGURE_DPI = 300


def plot_combined(
    fidas: dict,
    ucass: dict,
    tsi: dict,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.loglog(
        fidas["Diameter_um"],
        mask_nonpositive_for_log(fidas["dN_dlnDp"]),
        "-o",
        markersize=3,
        linewidth=1.2,
        color="#9467bd",
        label="FIDAS 200",
        zorder=5,
    )

    for uid in UCASS_IDS:
        r = ucass[uid]
        ax.loglog(
            r["Diameter_um"],
            mask_nonpositive_for_log(r["dN_dlnDp"]),
            "-o",
            markersize=5,
            linewidth=1.2,
            color=UCASS_COLORS[uid],
            label=f"UCASS {uid}",
        )

    ax.loglog(
        tsi["Diameter_um"],
        mask_nonpositive_for_log(tsi["dN_dlnDp"]),
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

    tsi_spectra, _, tsi_bins = load_tsi_spectra()
    master = build_ucass_master()
    fidas_df = load_fidas()
    reference, period_start, period_end = select_reference_period(master)

    cal = load_calibrations()
    ucass_bins = {uid: cal[CALIBRATION_MAP[uid]]["centre"][1:] for uid in UCASS_IDS}

    ucass_results, _ = compute_ucass_overlap_psd(
        master, ucass_bins, reference,
    )
    fidas_results, _ = compute_fidas_overlap_psd(
        fidas_df, reference,
    )
    tsi_results, _ = compute_tsi_overlap_psd(
        tsi_spectra, tsi_bins, reference,
    )

    print("Combined FIDAS / UCASS / TSI dN/dlnDp overlay (UCASS-matched timestamps)")
    print("=" * 60)
    print(f"UCASS reference timestamps : {len(reference)}")
    print(f"Period bounds            : {period_start} -> {period_end} UTC")
    print(f"FIDAS spectra  : {fidas_results['n_records']}")
    print(f"TSI spectra    : {tsi_results['n_records']}")
    for uid in UCASS_IDS:
        print(f"UCASS {uid} records : {ucass_results[uid]['n_records']}")

    plot_combined(
        fidas_results,
        ucass_results,
        tsi_results,
        period_start,
        period_end,
        OUTPUT_PNG,
    )
    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
