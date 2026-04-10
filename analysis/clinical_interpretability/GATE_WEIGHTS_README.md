# Gate Weight Extraction and Brain Visualization Workflow

This directory contains scripts for extracting gate weights from trained **mref-ad** (MoE) models and visualizing them on brain maps.

## Quick Start

### Option 1: Extract from Full Model Runs (Recommended)

If you have already trained models with multiple seeds and saved gate weights:

```bash
# Extract gate weights and generate CSV for brain plotting
bash extract_full_model_gates.sh results results/gate_weights_brain.csv

# This will create:
# - results/gate_weights_brain.csv (for R brain plotting)
# - results/gate_weights_brain_summary.csv (full expert list)
```

### Option 2: Extract from Existing JSON Results

If you have JSON results from cross-validation or hierarchical models:

```bash
# Run the plotting script which will also export CSVs
python3 scripts/plot_moe_interpretability.py

# This will create:
# - results/plots/moe_gate_weights_for_brain_plot.csv (for R brain plotting)
# - results/plots/moe_gate_heatmap_values.csv (fold-level data)
# - results/plots/*.png (visualization figures)
```

## Workflow Details

### 1. Training Models with Gate Weight Extraction

The gate weight extraction has been integrated into `train_moe.py`:

- **During retrain phase**: Gate weights are automatically saved as `*_full_gates.npy` files
- **Location**: Same directory as model checkpoints (e.g., `results/moe_seed42_full_gates.npy`)
- **Format**: NumPy array with shape `(n_samples, n_experts)`

To enable this, models must be trained with the retrain phase:
```bash
# Example: Train with multiple seeds
bash run_ablation_seeds.sh
```

### 2. Extracting Gate Weights

#### From .npy Files (Recommended)

```bash
python3 scripts/extract_full_model_gates.py \
    --results_dir results \
    --output results/gate_weights_brain.csv \
    --summary_output results/gate_weights_summary.csv
```

The script will:
1. Find all `*_full_gates.npy` files in the results directory
2. Load gate weights from each seed
3. Aggregate across all samples and seeds
4. Generate CSV files in the format needed for brain plotting

#### From JSON Results Files

```bash
python3 scripts/plot_moe_interpretability.py
```

This processes JSON results and generates:
- Visualization figures
- CSV exports for brain plotting
- Fold-level statistics

### 3. Brain Visualization in R

Once you have the CSV file, use it with the R brain plotting script:

```bash
# Open the R script
# Update the path in line ~25:
# df_plot <- as.data.frame(fread("results/gate_weights_brain.csv", ...))

# Run the script
Rscript scripts/brain_plot_clean_zixuan.R
```

Expected CSV format:
```csv
ggseg_dk,mri,amy,dk
Frontal Lobe,0.0523,0.0489,1
Temporal Lobe,0.0612,0.0701,1
Subcortical Temporal,0.0234,0.0198,0
...
```

Columns:
- `ggseg_dk`: Region name (compatible with ggseg R package)
- `mri`: MRI expert weight for this region
- `amy`: Amyloid PET expert weight for this region
- `dk`: 1 for cortical (DK atlas), 0 for subcortical (ASEG atlas)

## File Structure

```
multimodal-imaging-agents/
├── scripts/
│   ├── train_moe.py                      # Training script (saves gate weights)
│   ├── extract_full_model_gates.py       # Extract gates from .npy files
│   ├── analyze_gate_weights.py           # Analyze gates (for ablations)
│   ├── plot_moe_interpretability.py      # Visualization + CSV export
│   └── brain_plot_clean_zixuan.R         # R brain visualization
├── extract_full_model_gates.sh           # Bash wrapper for extraction
├── extract_gate_weights.sh               # Extract gates from ablations
└── run_ablation_and_analyze.sh           # Full ablation workflow
```

## Output Files

### Gate Weight Files (.npy)
- Location: `results/*_full_gates.npy`
- Format: NumPy array `(n_samples, n_experts)`
- Created by: `train_moe.py` during retrain phase
- One file per seed

### Brain Plot CSVs
- **Primary**: `results/gate_weights_brain.csv`
- **Summary**: `results/gate_weights_brain_summary.csv`
- Format: CSV with columns `ggseg_dk, mri, amy, dk`
- Compatible with: `brain_plot_clean_zixuan.R`

### Analysis CSVs
- `results/plots/moe_gate_heatmap_values.csv`: Fold-level gate weights
- `results/plots/moe_gate_subject_heatmap_*.csv`: Subject-level analysis
- `results/gate_analysis/gate_weights_summary.json`: Full statistics

## Troubleshooting

### No gate weight files found
**Problem**: Script reports "No .npy gate weight files found"
**Solution**: 
1. Check that models were trained with retrain phase enabled
2. Verify files exist: `ls results/*_full_gates.npy`
3. Try using JSON input: `python3 scripts/plot_moe_interpretability.py`

### Missing expert names
**Problem**: Experts named as `expert_0, expert_1, ...`
**Solution**: 
1. Ensure JSON results exist in results directory
2. The script will try to infer expert names from JSON files

### R script fails to load data
**Problem**: `brain_plot_clean_zixuan.R` cannot find CSV
**Solution**:
1. Check CSV path in R script (line ~25)
2. Verify CSV format matches expected columns
3. Check that region names are compatible with ggseg atlas

## Advanced Usage

### Extracting Gates from Specific Seeds

```python
import numpy as np

# Load specific seed
gates_seed42 = np.load("results/moe_seed_42_full_gates.npy")
print(f"Shape: {gates_seed42.shape}")  # (n_samples, n_experts)

# Compute mean per expert
mean_weights = gates_seed42.mean(axis=0)
print(f"Mean weights: {mean_weights}")
```

### Analyzing Expert Selection Patterns

```bash
# Run analysis with statistics
python3 scripts/analyze_gate_weights.py \
    --ablation gate_region_only \
    --output results/gate_analysis

# This generates:
# - aggregated_gates.npy
# - gate_weights_summary.json (with entropy, diversity metrics)
```

### Comparing Different Model Configurations

```bash
# Extract gates from different configurations
python3 scripts/extract_full_model_gates.py --results_dir results/config1 --output results/gates_config1.csv
python3 scripts/extract_full_model_gates.py --results_dir results/config2 --output results/gates_config2.csv

# Compare in Python
import pandas as pd
df1 = pd.read_csv("results/gates_config1_summary.csv")
df2 = pd.read_csv("results/gates_config2_summary.csv")
```

## Related Scripts

- `run_ablation_seeds.sh`: Train models with multiple seeds
- `test_moe_topk.sh`: Test different topk constraints
- `aggregate_cv.py`: Aggregate cross-validation results
