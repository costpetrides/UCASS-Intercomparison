#!/usr/bin/env python3
"""
Compute and plot the mean particle size distribution (PSD) from a TSI OPS 3330
CSV export.

Reads per-bin raw counts, converts to number concentration (#/cm³) using the
OPS manual relation, averages all valid spectra, and writes a CSV plus figure.

Sized channels: Bin 1–16 (boundaries from TSI_OPS3330_bin_boundaries.xlsx).
Bin 17 is overflow (> last bin upper boundary) and is excluded from the PSD.

Dependencies: numpy, pandas, matplotlib, openpyxl
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TSI_DIR = ROOT / "data" / "tsi"
DEFAULT_BINS = TSI_DIR / "TSI_OPS3330_bin_boundaries.xlsx"

TSI_MULTI_FILES = ("TSI_1.csv", "TSI_2.csv")
TSI_LEGACY_FILES = ("TSI_OPS3330.csv", "TSI.csv")

OUTPUT_CSV = ROOT / "outputs" / "tsi" / "TSI_mean_PSD.csv"
OUTPUT_PNG = ROOT / "outputs" / "tsi" / "TSI_mean_PSD.png"

FLOW_RATE_CM3_S = 16.67  # 1.0 L/min sample flow (OPS 3330)
OVERFLOW_BIN = 17
FIGURE_DPI = 300


# ---------------------------------------------------------------------------
# Input resolution and loading
# ---------------------------------------------------------------------------

def resolve_tsi_csv_files() -> list[Path]:
    """Return TSI export CSV paths (multi-file campaign preferred)."""
    multi = [TSI_DIR / name for name in TSI_MULTI_FILES if (TSI_DIR / name).is_file()]
    if multi:
        return multi

    for name in TSI_LEGACY_FILES:
        path = TSI_DIR / name
        if path.is_file():
            return [path]

    expected = ", ".join(TSI_MULTI_FILES + TSI_LEGACY_FILES)
    raise FileNotFoundError(
        f"TSI CSV not found in {TSI_DIR} (expected one of: {expected})"
    )


def resolve_tsi_csv() -> Path:
    """Backward-compatible single-file resolver."""
    files = resolve_tsi_csv_files()
    if len(files) > 1:
        raise ValueError(
            "Multiple TSI CSV files are configured; use load_tsi_spectra() instead."
        )
    return files[0]


def parse_hms_interval(value: str) -> float:
    """Parse H:M:S or D:H:M:S interval string to seconds (uses trailing H:M:S)."""
    parts = value.strip().split(":")
    if len(parts) == 4:
        parts = parts[1:]
    if len(parts) != 3:
        raise ValueError(f"Cannot parse sample interval: {value!r}")
    hours, minutes, seconds = (int(parts[0]), int(parts[1]), float(parts[2]))
    return hours * 3600 + minutes * 60 + seconds


def parse_tsi_metadata(csv_path: Path) -> tuple[dict[str, str], int]:
    """Return metadata key-value pairs and the line index of the data header row."""
    meta: dict[str, str] = {}
    header_idx = None
    for idx, line in enumerate(csv_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        stripped = line.strip()
        if stripped.startswith("Elapsed Time"):
            header_idx = idx
            break
        if "," in stripped:
            key, value = stripped.split(",", 1)
            meta[key.strip()] = value.strip()
    if header_idx is None:
        raise ValueError(f"No data header row found in {csv_path}")
    return meta, header_idx


def load_tsi_bin_boundaries(path: Path = DEFAULT_BINS) -> pd.DataFrame:
    """Load lower/upper diameter boundaries (µm) from Excel."""
    if not path.is_file():
        raise FileNotFoundError(f"TSI bin boundary file not found: {path}")
    table = pd.read_excel(path)
    required = {"Bin", "Lower_um", "Upper_um"}
    if not required.issubset(table.columns):
        raise ValueError(f"Bin boundary file must contain columns {required}")
    table = table.sort_values("Bin").reset_index(drop=True)
    table["Diameter_um"] = np.sqrt(table["Lower_um"] * table["Upper_um"])
    return table


def load_tsi_spectra_from_file(
    csv_path: Path,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """Load one TSI export file into spectra, metadata, and bin boundaries."""
    meta, header_idx = parse_tsi_metadata(csv_path)

    raw = pd.read_csv(csv_path, skiprows=header_idx)
    raw.columns = [str(c).strip() for c in raw.columns]

    start_ts = pd.to_datetime(
        f"{meta['Test Start Date']} {meta['Test Start Time']}",
        format="%Y/%m/%d %H:%M:%S",
    )
    elapsed = pd.to_numeric(raw["Elapsed Time [s]"], errors="coerce")
    raw["Timestamp"] = start_ts + pd.to_timedelta(elapsed, unit="s")

    sample_interval_s = parse_hms_interval(meta["Sample Interval [H:M:S]"])
    dtc = float(meta["DeadTime Correction Factor"])

    bins = load_tsi_bin_boundaries()
    count_cols = [f"Bin {int(b)}" for b in bins["Bin"]]
    missing = [c for c in count_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected bin columns in {csv_path}: {missing}")

    counts = raw[count_cols].apply(pd.to_numeric, errors="coerce")
    deadtime = pd.to_numeric(raw["Deadtime (s)"], errors="coerce")
    effective_time = sample_interval_s - dtc * deadtime
    sample_vol = FLOW_RATE_CM3_S * effective_time

    valid = counts.notna().all(axis=1) & deadtime.notna() & (effective_time > 0)
    if not valid.any():
        raise ValueError(f"No valid TSI spectra in {csv_path} after filtering.")

    spectra = raw.loc[valid, ["Timestamp", "Elapsed Time [s]", "Deadtime (s)"]].copy()
    spectra["effective_time_s"] = effective_time.loc[valid].to_numpy()
    spectra["sample_vol_cm3"] = sample_vol.loc[valid].to_numpy()
    spectra["source_file"] = csv_path.name

    count_values = counts.loc[valid].to_numpy(dtype=float)
    conc = count_values / spectra["sample_vol_cm3"].to_numpy()[:, None]

    for i, col in enumerate(count_cols):
        spectra[col] = count_values[:, i]
        spectra[f"conc_{col.replace(' ', '_')}"] = conc[:, i]

    if f"Bin {OVERFLOW_BIN}" in raw.columns:
        overflow = pd.to_numeric(raw.loc[valid, f"Bin {OVERFLOW_BIN}"], errors="coerce")
        spectra[f"Bin {OVERFLOW_BIN}"] = overflow.to_numpy()
        spectra["overflow_conc_cm3"] = overflow.to_numpy() / spectra["sample_vol_cm3"].to_numpy()

    meta = dict(meta)
    meta["Source file"] = csv_path.name
    return spectra, meta, bins


def load_tsi_spectra(csv_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame]:
    """
    Load TSI export(s): metadata, sized-bin table, and per-sample concentrations.

    When csv_path is omitted, loads TSI_1.csv and TSI_2.csv when present and
    concatenates them in timestamp order.

    Returns
    -------
    spectra : DataFrame with Timestamp, bin counts, Ci columns, sample_vol_cm3
    meta    : metadata dictionary from file header(s)
    bins    : bin boundary table (16 sized channels)
    """
    if csv_path is not None:
        return load_tsi_spectra_from_file(csv_path)

    paths = resolve_tsi_csv_files()
    if len(paths) == 1:
        return load_tsi_spectra_from_file(paths[0])

    spectra_parts: list[pd.DataFrame] = []
    metas: list[dict[str, str]] = []
    bins: pd.DataFrame | None = None

    for path in paths:
        spectra, meta, bins_table = load_tsi_spectra_from_file(path)
        spectra_parts.append(spectra)
        metas.append(meta)
        bins = bins_table

    combined = (
        pd.concat(spectra_parts, ignore_index=True)
        .sort_values("Timestamp")
        .reset_index(drop=True)
    )

    combined_meta = dict(metas[0])
    combined_meta["Source files"] = ", ".join(path.name for path in paths)
    combined_meta["Protocol Name_Number"] = " + ".join(
        meta.get("Protocol Name_Number", path.name)
        for meta, path in zip(metas, paths)
    )

    return combined, combined_meta, bins


# ---------------------------------------------------------------------------
# Averaging and outputs
# ---------------------------------------------------------------------------

def compute_mean_psd(spectra: pd.DataFrame, bins: pd.DataFrame) -> tuple[pd.Series, int]:
    conc_cols = [f"conc_Bin_{int(b)}" for b in bins["Bin"]]
    conc_data = spectra[conc_cols].apply(pd.to_numeric, errors="coerce")
    valid = conc_data.dropna(how="any")
    if valid.empty:
        raise ValueError("No valid concentration spectra remain.")

    mean_conc = valid.mean(axis=0)
    mean_psd = pd.Series(
        mean_conc.values,
        index=bins["Diameter_um"].values,
        name="Mean_dN",
    )
    return mean_psd, len(valid)


def print_summary(bins: pd.DataFrame, n_valid: int, meta: dict[str, str]) -> None:
    print("TSI mean PSD — summary")
    print("=" * 40)
    print(f"Total number of PSD bins     : {len(bins)}")
    print(f"Minimum particle diameter    : {bins['Diameter_um'].min():.4f} µm")
    print(f"Maximum particle diameter    : {bins['Diameter_um'].max():.4f} µm")
    print(f"Number of valid spectra      : {n_valid}")
    if "Source files" in meta:
        print(f"Source files                 : {meta['Source files']}")
    elif "Source file" in meta:
        print(f"Source file                  : {meta['Source file']}")
    print(f"Sample interval              : {meta.get('Sample Interval [H:M:S]')}")
    print(f"DeadTime correction factor   : {meta.get('DeadTime Correction Factor')}")
    print("Sized bin columns:")
    for _, row in bins.iterrows():
        print(f"  Bin {int(row['Bin'])}  [{row['Lower_um']:.3f}, {row['Upper_um']:.3f}] µm")
    print(f"Overflow channel Bin {OVERFLOW_BIN} excluded from sized PSD (manual: > upper boundary).")
    print()


def plot_mean_psd(mean_psd: pd.Series, output_path: Path) -> None:
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
    pd.DataFrame({
        "Diameter_um": mean_psd.index,
        "Mean_dN": mean_psd.values,
    }).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_path: Path | None = None) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    spectra, meta, bins = load_tsi_spectra(input_path)
    mean_psd, n_valid = compute_mean_psd(spectra, bins)

    print_summary(bins, n_valid, meta)
    save_mean_psd_csv(mean_psd, OUTPUT_CSV)
    plot_mean_psd(mean_psd, OUTPUT_PNG)

    print(f"Saved CSV : {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_PNG}")


if __name__ == "__main__":
    csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(csv_file)
