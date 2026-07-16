#!/usr/bin/env bash
# Run the full analysis pipeline from the repository root.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x .venv/bin/python3 ]]; then
    PYTHON=".venv/bin/python3"
  else
    PYTHON="python3"
  fi
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Using: $PYTHON"
"$PYTHON" scripts/Intercomparison.py
"$PYTHON" scripts/fidas_mean_psd.py
"$PYTHON" scripts/fidas_mean_dN_dlnDp.py
"$PYTHON" scripts/ucass_mean_per_bin_psd.py
"$PYTHON" scripts/ucass_multi_bin_scatter_10min.py
"$PYTHON" scripts/ucass_multi_bin_scatter_5min.py
"$PYTHON" scripts/ucass_multi_bin_scatter_5min_by_wind_2ms.py
"$PYTHON" scripts/ucass2_vs_ucass6_bin_scatter.py
"$PYTHON" scripts/overlap_period_dN_dlnDp_comparison.py
"$PYTHON" scripts/tsi_mean_psd.py
"$PYTHON" scripts/tsi_mean_dN_dlnDp.py
"$PYTHON" scripts/plot_combined_dN_dlnDp_FIDAS_UCASS_TSI.py
"$PYTHON" scripts/mean_dN_dlnDp_by_wind_speed.py
"$PYTHON" scripts/mean_dN_dlnDp_by_wind_direction.py
"$PYTHON" scripts/overlap_period_dN_dlnDp_comparison_FIDAS_UCASS_TSI.py
"$PYTHON" scripts/overlap_mean_total_concentration.py
"$PYTHON" scripts/wind_meteorology_analysis.py

echo "All scripts completed successfully."
