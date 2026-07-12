#!/usr/bin/env python3
"""
Convert FIDAS per-bin number concentrations (#/cm³) to dN/dlnDp for comparison
with UCASS integrated number PSDs from Intercomparison.py.

FIDAS dN* columns are per-bin concentrations. Bin boundaries are inferred
geometrically from sorted bin-centre diameters (same approach as UCASS).

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
    geometric_bin_boundaries,
    load_fidas_excel,
)

from plot_utils import mask_nonpositive_for_log

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "fidas" / "FIDAS200.txt"
DEFAULT_SHEET = 0

OUTPUT_CSV = ROOT / "outputs" / "fidas" / "FIDAS_mean_dN_dlnDp.csv"
OUTPUT_PNG = ROOT / "outputs" / "fidas" / "FIDAS_mean_dN_dlnDp.png"

FIGURE_DPI = 300


# ---------------------------------------------------------------------------
# Geometric bin boundaries and ΔlnDp
# ---------------------------------------------------------------------------

def delta_ln_dp(lower_um: np.ndarray, upper_um: np.ndarray) -> np.ndarray:
    """ΔlnDp = ln(D_upper / D_lower) for each bin."""
    return np.log(upper_um / lower_um)


def per_bin_dN_dlnDp(dn_bin: np.ndarray, dln: np.ndarray) -> np.ndarray:
    """dN/dlnDp = per-bin concentration / ΔlnDp."""
    return dn_bin / dln


# ---------------------------------------------------------------------------
# Loading, averaging, validation
# ---------------------------------------------------------------------------

def print_boundary_audit(
    psd_table: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    dln: np.ndarray,
) -> None:
    print("Inferred bin boundaries and ΔlnDp")
    print("=" * 72)
    print(
        f"{'Column':<14} {'D_centre':>10} {'D_lower':>12} "
        f"{'D_upper':>12} {'ΔlnDp':>12}"
    )
    print("-" * 72)
    for i, row in psd_table.iterrows():
        print(
            f"{row['column']:<14} {row['Diameter_um']:10.4f} "
            f"{lower[i]:12.6f} {upper[i]:12.6f} {dln[i]:12.6f}"
        )
    print()


def validate_integration(
    psd_data: pd.DataFrame,
    dln: np.ndarray,
    cn: pd.Series,
) -> None:
    """
    Verify Σ (dN/dlnDp × ΔlnDp) reproduces Σ dN (and Cn) per timestamp.
    """
    dn = psd_data.to_numpy(dtype=float)
    dN_dlnDp = dn / dln
    reintegrated = (dN_dlnDp * dln).sum(axis=1)
    sum_dn = dn.sum(axis=1)

    diff_reint = reintegrated - sum_dn
    diff_cn = sum_dn - cn.to_numpy(dtype=float)

    print("Integration validation (per timestamp)")
    print("=" * 50)
    print(f"Timestamps validated : {len(psd_data)}")
    print(
        "Σ (dN/dlnDp × ΔlnDp) vs Σ dN — "
        f"max |diff| = {np.max(np.abs(diff_reint)):.6e}, "
        f"mean |diff| = {np.mean(np.abs(diff_reint)):.6e}"
    )
    print(
        "Σ dN vs Cn — "
        f"max |diff| = {np.max(np.abs(diff_cn)):.6f}, "
        f"mean |diff| = {np.mean(np.abs(diff_cn)):.6f}, "
        f"mean ratio ΣdN/Cn = {(sum_dn / cn).mean():.6f}"
    )

    # Mean spectrum integration check
    mean_dn = dn.mean(axis=0)
    mean_dN_dlnDp = mean_dn / dln
    sum_mean_reint = np.sum(mean_dN_dlnDp * dln)
    sum_mean_dn = np.sum(mean_dn)
    mean_cn = cn.mean()

    print()
    print("Mean-spectrum integration check")
    print("-" * 50)
    print(f"Σ (mean dN/dlnDp × ΔlnDp) : {sum_mean_reint:.6f} #/cm³")
    print(f"Σ mean dN                 : {sum_mean_dn:.6f} #/cm³")
    print(f"mean Cn                   : {mean_cn:.6f} #/cm³")
    print(
        f"|Σ mean dN − mean Cn|      : {abs(sum_mean_dn - mean_cn):.6f} #/cm³"
    )
    print(
        f"|reintegrated − Σ mean dN|: "
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
    ax.set_title("FIDAS mean number size distribution (dN/dlnDp)")
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

def main(input_path: Path = DEFAULT_INPUT) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_fidas_excel(input_path)
    df, period_start, period_end = filter_fidas_to_ucass_period(df)
    print(
        f"FIDAS restricted to UCASS period: "
        f"{period_start} -> {period_end} UTC ({len(df)} spectra)"
    )

    psd_columns = detect_psd_columns(df.columns)
    if not psd_columns:
        raise ValueError("No PSD columns found.")

    psd_table = build_psd_table(psd_columns)
    centres = psd_table["Diameter_um"].to_numpy(dtype=float)
    cols = psd_table["column"].tolist()

    lower, upper = geometric_bin_boundaries(centres)
    dln = delta_ln_dp(lower, upper)

    psd_data = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = psd_data.dropna(how="any")
    if valid.empty:
        raise ValueError("No valid spectra after NaN removal.")

    cn = pd.to_numeric(df.loc[valid.index, "Cn"], errors="coerce")

    print_boundary_audit(psd_table, lower, upper, dln)
    validate_integration(valid, dln, cn)

    mean_dn = valid.mean(axis=0).to_numpy(dtype=float)
    mean_dN_dlnDp = per_bin_dN_dlnDp(mean_dn, dln)

    save_csv(centres, mean_dN_dlnDp, OUTPUT_CSV)
    plot_mean_dN_dlnDp(centres, mean_dN_dlnDp, OUTPUT_PNG)

    print(f"Saved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")
    print(f"Valid spectra averaged: {len(valid)}")


if __name__ == "__main__":
    input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    main(input_file)
