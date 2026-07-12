# Project structure

```
Intercomparison/
├── scripts/          # Analysis and plotting scripts
├── data/
│   ├── ucass/        # UCASS CSV exports and size-bin calibrations
│   ├── fidas/        # FIDAS 200 Excel export
│   ├── tsi/          # TSI OPS 3330 CSV export and bin boundaries
│   └── wind/         # Campbell TOA5 wind (RTF)
├── outputs/
│   ├── ucass/        # Main UCASS intercomparison outputs
│   ├── ucass_mean/   # Standalone UCASS mean per-bin PSD
│   ├── fidas/        # Standalone FIDAS mean PSD outputs
│   ├── tsi/          # Standalone TSI mean PSD outputs
│   ├── overlap/      # Overlap-period comparison figures and CSVs
│   ├── combined/     # Combined overlay figures
│   └── wind/         # Wind meteorology analysis outputs
└── docs/             # Documentation
```

## Reference instruments (FIDAS and TSI)

Each reference instrument has an independent processing branch with parallel outputs:

| Instrument | Scripts | Outputs |
|------------|---------|---------|
| FIDAS 200 | `fidas_mean_psd.py`, `fidas_mean_dN_dlnDp.py` | `outputs/fidas/FIDAS_mean_PSD.*`, `FIDAS_mean_dN_dlnDp.*` |
| TSI OPS 3330 | `tsi_mean_psd.py`, `tsi_mean_dN_dlnDp.py` | `outputs/tsi/TSI_mean_PSD.*`, `TSI_mean_dN_dlnDp.*` |

Comparison figures (read pre-computed CSVs only) live in `outputs/combined/` and `outputs/overlap/`:

| Figure | Script |
|--------|--------|
| `combined_FIDAS_UCASS_TSI_dN_dlnDp.png` | `plot_combined_dN_dlnDp_FIDAS_UCASS_TSI.py` |
| `combined/wind_speed/mean_dN_dlnDp_WS_*.png` | `mean_dN_dlnDp_by_wind_speed.py` |
| `combined/wind_direction/mean_dN_dlnDp_WD_*.png` | `mean_dN_dlnDp_by_wind_direction.py` |
| `overlap_period_FIDAS_UCASS_dN_dlnDp.png` | `overlap_period_dN_dlnDp_comparison.py` |
| `overlap_period_FIDAS_UCASS_TSI_dN_dlnDp.png` | `overlap_period_dN_dlnDp_comparison_FIDAS_UCASS_TSI.py` |
| `mean_total_particle_concentration.png` | `overlap_mean_total_concentration.py` |

Run scripts from the repository root, for example:

```bash
# Recommended: use the project venv (has windrose, scipy, etc.)
source .venv/bin/activate
python3 scripts/Intercomparison.py

# Or run everything at once:
./run_all.sh
```

If you use system `python3` instead of `.venv`, install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```
