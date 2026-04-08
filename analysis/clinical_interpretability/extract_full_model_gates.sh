#!/bin/bash
################################################################################
# Extract gate weights from full model and generate CSV for brain plotting
#
# This script:
# 1. Finds gate weight files from full model runs (all seeds)
# 2. Aggregates gate weights across seeds
# 3. Generates CSV files for brain visualization in R
#
# Usage: bash extract_full_model_gates.sh [results_dir] [output_csv]
# Example: bash extract_full_model_gates.sh results results/gate_weights_brain.csv
################################################################################

set -e

RESULTS_DIR=${1:-"results"}
OUTPUT_CSV=${2:-"results/gate_weights_for_brain_plot.csv"}
SUMMARY_CSV="${OUTPUT_CSV%.csv}_summary.csv"

echo "=================================="
echo "EXTRACT GATE WEIGHTS FROM FULL MODEL"
echo "=================================="
echo "Results directory: $RESULTS_DIR"
echo "Output CSV: $OUTPUT_CSV"
echo "Summary CSV: $SUMMARY_CSV"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "[ERROR] Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Run extraction script
python3 scripts/extract_full_model_gates.py \
    --results_dir "$RESULTS_DIR" \
    --output "$OUTPUT_CSV" \
    --summary_output "$SUMMARY_CSV"

# Check if output was generated
if [ -f "$OUTPUT_CSV" ]; then
    echo ""
    echo "========== SUCCESS =========="
    echo "Gate weights extracted successfully!"
    echo ""
    echo "Output files:"
    echo "  1. Brain plot CSV: $OUTPUT_CSV"
    echo "  2. Summary CSV: $SUMMARY_CSV"
    echo ""
    echo "Preview of brain plot CSV:"
    head -10 "$OUTPUT_CSV"
    echo ""
    echo "Next steps:"
    echo "  1. Open scripts/brain_plot_clean_zixuan.R"
    echo "  2. Update YOUR_ROI_DATA_ALL.csv path to: $OUTPUT_CSV"
    echo "  3. Run: Rscript scripts/brain_plot_clean_zixuan.R"
else
    echo ""
    echo "[ERROR] Failed to generate output CSV"
    exit 1
fi
