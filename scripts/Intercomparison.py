import re 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "ucass"

UCASS27_CSV = ROOT / "data" / "ucass" / "UCASS13.csv"
UCASS362_CSV = ROOT / "data" / "ucass" / "UCASS62.csv"
BINS_NUMBERS = ROOT / "data" / "ucass" / "UCASS_size_bins.numbers"
BINS_XLSX = ROOT / "data" / "ucass" / "UCASS_size_bins.xlsx"
WIND_RTF = ROOT / "data" / "wind" / "wind.rtf"

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

SAMPLE_AREA = 5.0e-07          # m²
PARTICLE_DENSITY = 2.6         # g/cm³
UCASS_DATE_FORMAT = "%d/%m/%y %H:%M:%S"

CALIBRATION_MAP = {
    1: "AA001",
    6: "AA006",
    2: "AD002",
    5: "AD005",
    # Temporary until confirmed
    3: "AA001",
}

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


# ============================================================
# LOAD UCASS CALIBRATIONS
# ============================================================

def load_bins_table(bins_file):

    if Path(bins_file).suffix.lower() == ".xlsx":
        return pd.read_excel(bins_file, header=None)

    from numbers_parser import Document

    doc = Document(bins_file)
    table = doc.sheets[0].tables[0]

    rows = []
    for row_idx in range(table.num_rows):
        rows.append([
            table.cell(row_idx, col_idx).value
            for col_idx in range(table.num_cols)
        ])

    return pd.DataFrame(rows)


def resolve_bins_file():
    if BINS_NUMBERS.is_file():
        return BINS_NUMBERS
    if BINS_XLSX.is_file():
        return BINS_XLSX
    raise FileNotFoundError(
        f"Calibration file not found: {BINS_NUMBERS} or {BINS_XLSX}"
    )


def load_calibrations(bins_file=None):

    if bins_file is None:
        bins_file = resolve_bins_file()

    raw = load_bins_table(bins_file)

    calibrations = {}

    mapping = {
        "AA001": (0, 1),
        "AA006": (2, 3),
        "AD002": (4, 5),
        "AD005": (6, 7),
    }

    for name, (col_low, col_up) in mapping.items():

        lower = pd.to_numeric(
            raw.iloc[2:, col_low],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        upper = pd.to_numeric(
            raw.iloc[2:, col_up],
            errors="coerce"
        ).dropna().to_numpy(dtype=float)

        # Geometric centre (Alkistis Python implementation)
        centre = np.sqrt(lower * upper)

        calibrations[name] = {
            "lower": lower,
            "upper": upper,
            "centre": centre,
        }

    return calibrations


# ============================================================
# LOAD UCASS
# ============================================================

def load_ucass_csv(ucass_file, sep=","):

    df = pd.read_csv(
        ucass_file,
        skiprows=4,
        sep=sep,
        low_memory=False
    )

    df["Timestamp"] = pd.to_datetime(
        df["GPS_Date"].astype(str) + " " +
        df["GPS_Time[UTC]"].astype(str),
        format=UCASS_DATE_FORMAT,
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


def extract_ucass_counts(df, ucass_id, id_col="UCASS_ID", bin_suffix=""):

    raw_count_cols = [f"b{i}{bin_suffix}" for i in range(1, 16)]
    count_cols = [f"b{i}" for i in range(1, 16)]

    subset = df.loc[df[id_col] == ucass_id, ["Timestamp"] + raw_count_cols].copy()
    subset = subset.rename(
        columns={raw: std for raw, std in zip(raw_count_cols, count_cols)}
    )

    n_before = len(subset)
    n_dup_rows = subset["Timestamp"].duplicated().sum()

    if n_dup_rows:
        print(
            f"  UCASS ID {ucass_id}: {n_dup_rows} duplicate timestamp rows "
            f"-> summing bin counts per second"
        )
        subset = (
            subset.groupby("Timestamp", as_index=False)[count_cols]
            .sum()
        )

    subset = subset.sort_values("Timestamp").reset_index(drop=True)

    if len(subset) != n_before - n_dup_rows:
        raise ValueError(
            f"Unexpected row count after deduplicating UCASS ID {ucass_id}"
        )

    return subset


# ============================================================
# LOAD WIND (RTF-wrapped TOA5)
# ============================================================

def load_wind_rtf(wind_file):

    rtf_text = wind_file.read_text(encoding="utf-8", errors="replace")

    rows = []
    for raw_line in rtf_text.split("\n"):
        line = raw_line.strip().rstrip("\\")
        match = WIND_DATA_PATTERN.search(line)
        if match:
            rows.append(match.groups())

    if not rows:
        raise ValueError(f"No wind data rows found in {wind_file}")

    df = pd.DataFrame(
        rows,
        columns=[
            "TIMESTAMP",
            "RECORD",
            "AirTC_Avg",
            "RH",
            "WindDir",
            "WS_ms_Avg",
            "WSDiag",
            "BP_mbar_Avg",
        ]
    )

    df["Timestamp"] = pd.to_datetime(
        df["TIMESTAMP"],
        errors="coerce"
    )

    numeric_cols = [
        "RECORD",
        "AirTC_Avg",
        "RH",
        "WindDir",
        "WS_ms_Avg",
        "WSDiag",
        "BP_mbar_Avg",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    return df


# ============================================================
# MERGE UCASS + WIND
# ============================================================

def merge_multi_ucass_wind(ucass_frames, df_wind):

    wind_cols = [
        "Timestamp",
        "WS_ms_Avg",
        "WindDir",
        "AirTC_Avg",
        "RH",
        "BP_mbar_Avg",
    ]

    print("\nTimestamp sync validation:")
    timestamp_sets = {}
    for uid, df in ucass_frames:
        timestamp_sets[uid] = set(df["Timestamp"])
        cfg = UCASS_SOURCES[uid]
        print(
            f"  UCASS {uid} records ({cfg['csv'].name}):",
            len(df),
        )

    common_ucass = set.intersection(*timestamp_sets.values())
    print("  Common UCASS timestamps (all instruments):", len(common_ucass))

    for uid, timestamps in timestamp_sets.items():
        print(
            f"  UCASS {uid} without full UCASS overlap:",
            len(timestamps - common_ucass),
        )

    master = None
    for uid, df in ucass_frames:
        renamed = df.rename(
            columns={f"b{i}": f"b{i}_id{uid}" for i in range(1, 16)}
        )
        if master is None:
            master = renamed
        else:
            master = pd.merge(
                master,
                renamed,
                on="Timestamp",
                how="inner",
                validate="one_to_one",
            )

    print("  Merged UCASS rows (inner join):", len(master))

    wind_timestamps = set(df_wind["Timestamp"])
    master_timestamps = set(master["Timestamp"])
    wind_intersection = master_timestamps & wind_timestamps

    print("  Wind records (full file):", len(df_wind))
    print("  Merged UCASS rows with wind:", len(wind_intersection))
    print(
        "  Merged UCASS rows without wind:",
        len(master_timestamps - wind_timestamps),
    )

    if len(wind_intersection) != len(master):
        raise ValueError(
            "Wind timestamp mismatch after UCASS merge: "
            f"{len(wind_intersection)} matches for {len(master)} merged rows"
        )

    master = pd.merge(
        master,
        df_wind[wind_cols],
        on="Timestamp",
        how="inner",
        validate="one_to_one",
    )

    if len(master) != len(wind_intersection):
        raise ValueError(
            "Inner merge row count mismatch: "
            f"{len(master)} merged rows for {len(wind_intersection)} matches"
        )

    print("  Merge OK: one-to-one UCASS/wind match at 1 Hz")
    print(
        "  Overlap:",
        master["Timestamp"].min(),
        "->",
        master["Timestamp"].max(),
    )

    return master


# ============================================================
# LOAD FILES
# ============================================================

df_wind = load_wind_rtf(WIND_RTF)
cal = load_calibrations()

raw_cache = {}
ucass_frames = []

for uid in UCASS_IDS:
    cfg = UCASS_SOURCES[uid]
    cache_key = (cfg["csv"], cfg["sep"])
    if cache_key not in raw_cache:
        raw_cache[cache_key] = load_ucass_csv(cfg["csv"], sep=cfg["sep"])

    df_counts = extract_ucass_counts(
        raw_cache[cache_key],
        uid,
        id_col=cfg["id_col"],
        bin_suffix=cfg["bin_suffix"],
    )
    ucass_frames.append((uid, df_counts))

    print(f"UCASS {uid} rows ({cfg['csv'].name}):", len(df_counts))
    print(
        f"UCASS {uid} time range:",
        df_counts["Timestamp"].min(),
        "->",
        df_counts["Timestamp"].max(),
    )

print("Wind rows:", len(df_wind))
print(
    "Wind time range:",
    df_wind["Timestamp"].min(),
    "->",
    df_wind["Timestamp"].max(),
)


# ============================================================
# UCASS CALIBRATIONS
# ============================================================

ucass_cal = {uid: CALIBRATION_MAP[uid] for uid in UCASS_IDS}

for uid in UCASS_IDS:
    print(f"UCASS {uid} calibration =", ucass_cal[uid])


# ============================================================
# BIN CENTRES
# ============================================================

ucass_bins = {
    uid: cal[ucass_cal[uid]]["centre"][1:]
    for uid in UCASS_IDS
}

for uid in UCASS_IDS:
    print(f"\nUCASS {uid} bins =", len(ucass_bins[uid]))


# ============================================================
# dlnDp
# ============================================================

dlnDp = {}
for uid in UCASS_IDS:
    centres = ucass_bins[uid]
    widths = np.log(centres[1:]) - np.log(centres[:-1])
    dlnDp[uid] = np.append(widths, widths[-1])


# ============================================================
# MERGE UCASS + WIND (1 Hz)
# ============================================================

master = merge_multi_ucass_wind(ucass_frames, df_wind)

print(
    master[
        [
            "Timestamp",
            "WS_ms_Avg",
            "AirTC_Avg",
            "RH",
        ]
    ].head()
)


# ============================================================
# SAMPLE VOLUME (1 second)
# ============================================================

master["sample_vol_cm3"] = (
    SAMPLE_AREA *
    master["WS_ms_Avg"] *
    1e6
)

print("\nSample volume statistics [cm³]:")
print(master["sample_vol_cm3"].describe())

sample_vol = master["sample_vol_cm3"].to_numpy(dtype=float)
total_sample_vol = sample_vol.sum()


# ============================================================
# CONCENTRATION + PSD + TOTALS
# ============================================================

ucass_counts = {}
conc = {}
conc_col = {}
dNdlnDp = {}
dVdlnDp = {}
dMdlnDp = {}
part_vol = {}
part_mass = {}

for uid in UCASS_IDS:
    count_cols = [f"b{i}_id{uid}" for i in range(1, 16)]
    counts = master[count_cols].to_numpy(dtype=float)
    concentrations = counts / sample_vol[:, None]

    ucass_counts[uid] = counts
    conc[uid] = concentrations
    conc_col[uid] = counts.sum(axis=0) / total_sample_vol

    dNdlnDp[uid] = conc_col[uid] / dlnDp[uid]

    bin_vol = (np.pi / 6.0) * (ucass_bins[uid] ** 3)
    dVdlnDp[uid] = conc_col[uid] * bin_vol / dlnDp[uid]
    dMdlnDp[uid] = dVdlnDp[uid] * PARTICLE_DENSITY

    part_vol[uid] = concentrations * bin_vol
    part_mass[uid] = part_vol[uid] * PARTICLE_DENSITY

    master[f"UCASS{uid}_total_counts"] = counts.sum(axis=1)
    master[f"UCASS{uid}_total_number_cm3"] = np.sum(concentrations, axis=1)
    master[f"UCASS{uid}_total_volume_um3_cm3"] = np.sum(part_vol[uid], axis=1)
    master[f"UCASS{uid}_total_mass_ug_m3"] = np.sum(part_mass[uid], axis=1)

    print(f"\nUCASS {uid} concentration shape:", concentrations.shape)
    print(f"UCASS {uid} total number concentration")
    print(master[f"UCASS{uid}_total_number_cm3"].describe())


# ============================================================
# EXPORT PSD
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

psd_paths = []
for uid in UCASS_IDS:
    psd = pd.DataFrame({
        "Dp_um": ucass_bins[uid],
        "dN_dlnDp_cm3": dNdlnDp[uid],
        "dV_dlnDp_um3_cm3": dVdlnDp[uid],
        "dM_dlnDp_ug_m3": dMdlnDp[uid],
    })
    psd_path = OUTPUT_DIR / f"UCASS{uid}_integrated_psd.csv"
    psd.to_csv(psd_path, index=False)
    psd_paths.append(psd_path)

print("\nSaved:")
for path in psd_paths:
    print(path)


# ============================================================
# EXPORT TOTALS
# ============================================================

# ============================================================
# PLOTS
# ============================================================

# TOTAL COUNTS VS TIME
plt.figure(figsize=(10, 5))
for uid in UCASS_IDS:
    plt.plot(
        master["Timestamp"],
        master[f"UCASS{uid}_total_counts"],
        "-",
        label=f"UCASS {uid}",
        linewidth=0.8,
    )
plt.xlabel("Time")
plt.ylabel("Total Counts (particles s$^{-1}$)")
plt.title("Total Particle Counts")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_counts_vs_time.png", dpi=300)
plt.close()

# TOTAL NUMBER VS TIME
plt.figure(figsize=(10, 5))
for uid in UCASS_IDS:
    plt.plot(
        master["Timestamp"],
        master[f"UCASS{uid}_total_number_cm3"],
        "-",
        label=f"UCASS {uid}",
        linewidth=0.8,
    )
plt.xlabel("Time")
plt.ylabel("Total Number Concentration (cm$^{-3}$)")
plt.title("Total Number Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_number_vs_time.png", dpi=300)
plt.close()

# TOTAL MASS VS TIME
plt.figure(figsize=(10, 5))
for uid in UCASS_IDS:
    plt.plot(
        master["Timestamp"],
        master[f"UCASS{uid}_total_mass_ug_m3"],
        "-",
        label=f"UCASS {uid}",
        linewidth=0.8,
    )
plt.xlabel("Time")
plt.ylabel("Total Mass (μg m$^{-3}$)")
plt.title("Total Mass Concentration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_mass_vs_time.png", dpi=300)
plt.close()

print("\nSaved plots:")
for plot_name in (
    "total_counts_vs_time.png",
    "total_number_vs_time.png",
    "total_mass_vs_time.png",
):
    print(OUTPUT_DIR / plot_name)
