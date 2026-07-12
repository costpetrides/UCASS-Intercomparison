#!/usr/bin/env python3
"""
Convert TSI OPS 3330 per-bin number concentrations (#/cm³) to dN/dlnDp for
comparison with UCASS integrated number PSDs from Intercomparison.py.

Bin boundaries are read explicitly from TSI_OPS3330_bin_boundaries.xlsx
(unlike FIDAS, where boundaries are inferred geometrically from centres).

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

from plot_utils import mask_nonpositive_for_log
from tsi_mean_psd import load_tsi_spectra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_CSV = ROOT / "outputs" / "tsi" / "TSI_mean_dN_dlnDp.csv"
OUTPUT_PNG = ROOT / "outputs" / "tsi" / "TSI_mean_dN_dlnDp.png"

FIGURE_DPI = 300


# ---------------------------------------------------------------------------
# ΔlnDp and dN/dlnDp
# ---------------------------------------------------------------------------

def delta_ln_dp(lower_um: np.ndarray, upper_um: np.ndarray) -> np.ndarray:
    """ΔlnDp = ln(D_upper / D_lower) for each bin."""
    return np.log(upper_um / lower_um)


def per_bin_dN_dlnDp(dn_bin: np.ndarray, dln: np.ndarray) -> np.ndarray:
    """dN/dlnDp = per-bin concentration / ΔlnDp."""
    return dn_bin / dln


# ---------------------------------------------------------------------------
# Validation (mirrors fidas_mean_dN_dlnDp.py)
# ---------------------------------------------------------------------------

def print_boundary_audit(
    bins: pd.DataFrame,
    dln: np.ndarray,
) -> None:
    print("TSI bin boundaries and ΔlnDp")
    print("=" * 72)
    print(
        f"{'Bin':<6} {'D_centre':>10} {'D_lower':>12} "
        f"{'D_upper':>12} {'ΔlnDp':>12}"
    )
    print("-" * 72)
    for i, row in bins.iterrows():
        print(
            f"{int(row['Bin']):<6} {row['Diameter_um']:10.4f} "
            f"{row['Lower_um']:12.6f} {row['Upper_um']:12.6f} {dln[i]:12.6f}"
        )
    print()


def validate_integration(
    conc_data: pd.DataFrame,
    dln: np.ndarray,
) -> None:
    """
    Verify Σ (dN/dlnDp × ΔlnDp) reproduces Σ Ci per timestamp.

    TSI has no separate Cn column; total concentration is sum of sized bins.
    """
    dn = conc_data.to_numpy(dtype=float)
    dN_dlnDp = dn / dln
    reintegrated = (dN_dlnDp * dln).sum(axis=1)
    sum_dn = dn.sum(axis=1)

    diff_reint = reintegrated - sum_dn

    print("Integration validation (per timestamp)")
    print("=" * 50)
    print(f"Timestamps validated : {len(conc_data)}")
    print(
        "Σ (dN/dlnDp × ΔlnDp) vs Σ Ci — "
        f"max |diff| = {np.max(np.abs(diff_reint)):.6e}, "
        f"mean |diff| = {np.mean(np.abs(diff_reint)):.6e}"
    )

    mean_dn = dn.mean(axis=0)
    mean_dN_dlnDp = mean_dn / dln
    sum_mean_reint = np.sum(mean_dN_dlnDp * dln)
    sum_mean_dn = np.sum(mean_dn)
    mean_total = sum_dn.mean()

    print()
    print("Mean-spectrum integration check")
    print("-" * 50)
    print(f"Σ (mean dN/dlnDp × ΔlnDp) : {sum_mean_reint:.6f} #/cm³")
    print(f"Σ mean Ci                 : {sum_mean_dn:.6f} #/cm³")
    print(f"mean Σ Ci per spectrum    : {mean_total:.6f} #/cm³")
    print(
        f"|reintegrated − Σ mean Ci|: "
        f"{abs(sum_mean_reint - sum_mean_dn):.6e} #/cm³"
    )
    print()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def plot_mean_dN_dlnDp(
    diameters: np.ndarray,
    dN_dlnDp: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(
        diameters,
        mask_nonpositive_for_log(dN_dlnDp),
        "-o",
        markersize=4,
        linewidth=1.2,
        color="steelblue",
    )

    ax.set_xlabel("Particle Diameter (µm)")
    ax.set_ylabel("dN/dlnDp (# cm$^{-3}$)")
    ax.set_title("TSI mean number size distribution (dN/dlnDp)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def save_csv(diameters: np.ndarray, dN_dlnDp: np.ndarray, path: Path) -> None:
    pd.DataFrame({
        "Diameter_um": diameters,
        "dN_dlnDp": dN_dlnDp,
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path | None = None) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    spectra, meta, bins = load_tsi_spectra(input_path)

    centres = bins["Diameter_um"].to_numpy(dtype=float)
    lower = bins["Lower_um"].to_numpy(dtype=float)
    upper = bins["Upper_um"].to_numpy(dtype=float)
    dln = delta_ln_dp(lower, upper)

    conc_cols = [f"conc_Bin_{int(b)}" for b in bins["Bin"]]
    conc_data = spectra[conc_cols].apply(pd.to_numeric, errors="coerce")
    valid = conc_data.dropna(how="any")
    if valid.empty:
        raise ValueError("No valid spectra after NaN removal.")

    print_boundary_audit(bins, dln)
    validate_integration(valid, dln)

    mean_dn = valid.mean(axis=0).to_numpy(dtype=float)
    mean_dN_dlnDp = per_bin_dN_dlnDp(mean_dn, dln)

    save_csv(centres, mean_dN_dlnDp, OUTPUT_CSV)
    plot_mean_dN_dlnDp(centres, mean_dN_dlnDp, OUTPUT_PNG)

    print(f"Saved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Valid spectra averaged: {len(valid)}")


if __name__ == "__main__":
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(csv_file)
