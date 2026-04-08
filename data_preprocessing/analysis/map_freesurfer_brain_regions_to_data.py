"""
map_freesurfer_brain_regions_to_data.py
---------------------------------------
This script maps Freesurfer brain region measures (from freesurfer_brain_regions.csv)
to column names in multimodal imaging datasets (e.g., Amyloid SUVR, MRI Volume).

It produces a mapping CSV file that shows which dataset columns correspond to
each Freesurfer region, which can then be used to define experts.yaml.

Example usage:
    $ conda activate ad-moe
    $ cd analysis
    $ python analysis/map_freesurfer_brain_regions_to_data.py \
        --regions data/freesurfer_brain_regions.csv \
        --data data/250826_DX_AMYLOID_multi_visit.csv \
        --output data/freesurfer_to_column_mapping.csv
"""

import pandas as pd
import re
import argparse

# --------------------------
# 1. Parse arguments
# --------------------------
parser = argparse.ArgumentParser(description="Map Freesurfer regions to data columns.")
parser.add_argument("--regions", type=str, required=True, help="Path to freesurfer_brain_regions.csv")
parser.add_argument("--data", type=str, required=True, help="Path to imaging dataset (CSV)")
parser.add_argument("--output", type=str, default="freesurfer_to_column_mapping.csv", help="Output CSV path")
args = parser.parse_args()

# --------------------------
# 2. Load input files
# --------------------------
regions = pd.read_csv(args.regions)
data = pd.read_csv(args.data)
data_cols = data.columns.str.upper()

print(f"Loaded {args.regions} ({len(regions)} regions)")
print(f"Loaded {args.data} with {len(data_cols)} columns")

# --------------------------
# 3. Define mapping patterns for Freesurfer short names → dataset naming
# --------------------------
mapping_patterns = {
    "AMYGVOL": ["AMYGDALA"],
    "HIPPVOL": ["HIPPOCAMPUS"],
    "ACCUMVOL": ["ACCUMBENS_AREA"],
    "CAUDVOL": ["CAUDATE"],
    "PALLVOL": ["PALLIDUM"],
    "PUTAMVOL": ["PUTAMEN"],
    "THALVOL": ["THALAMUS_PROPER"],
    "CEREBCTX": ["CEREBRAL_CORTEX"],
    "CEREBWM": ["CEREBRAL_WHITE_MATTER"],
    "CEREBELLCTX": ["CEREBELLUM_CORTEX"],
    "CEREBELLWM": ["CEREBELLUM_WHITE_MATTER"],
    "INFLATVENT": ["INF_LAT_VENT", "INFERIOR_LATERAL_VENTRICLE"],
    "LATVENT": ["LATERAL_VENTRICLE"],
    "CSF": ["CSF"],
    "CC_POST": ["CC_POSTERIOR"],
    "CC_MIDPOST": ["CC_MID_POSTERIOR"],
    "CC_CENT": ["CC_CENTRAL"],
    "CC_MIDANT": ["CC_MID_ANTERIOR"],
    "CC_ANT": ["CC_ANTERIOR"],
    "BRAINSTEM": ["BRAINSTEM", "BRAIN_STEM"],
    "PARSOPER": ["PARSOPERCULARIS"],
    "PARSORB": ["PARSORBITALIS"],
    "PARSTRIANG": ["PARSTRIANGULARIS"],
    "CAUDMIDFRONTAL": ["CAUDALMIDDLEFRONTAL"],
    "LATORBFRONTAL": ["LATERALORBITOFRONTAL"],
    "MEDORBFRONTAL": ["MEDIALORBITOFRONTAL"],
    "ROSTMIDFRONTAL": ["ROSTRALMIDDLEFRONTAL"],
    "SUPFRONTAL": ["SUPERIORFRONTAL"],
    "CAUDANTCING": ["CAUDALANTERIORCINGULATE"],
    "ISTHMCING": ["ISTHMUSCINGULATE"],
    "POSTCING": ["POSTERIORCINGULATE"],
    "ROSTANTCING": ["ROSTRALANTERIORCINGULATE"],
    "INFPARIETAL": ["INFERIORPARIETAL"],
    "SUPPARIETAL": ["SUPERIORPARIETAL"],
    "ENTCTX": ["ENTORHINAL"],
    "INFTEMPORAL": ["INFERIORTEMPORAL"],
    "MIDTEMPORAL": ["MIDDLETEMPORAL"],
    "SUPTEMPORAL": ["SUPERIORTEMPORAL"],
    "TRANSVTEMPORAL": ["TRANSVERSETEMPORAL"],
    "LATOCCIPITAL": ["LATERALOCCIPITAL"],
    "PRECUNEUS": ["PRECUNEUS"],
}

# --------------------------
# 4. Match each region to dataset columns
# --------------------------
region_matches = {}

for _, row in regions.iterrows():
    region = str(row["Measure"]).upper()
    patterns = mapping_patterns.get(region, [region])
    regex = "|".join([re.escape(p) for p in patterns])

    matched = [
        col for col in data_cols
        if re.search(regex, col)
        and (col.endswith("_SUVR") or col.endswith("_VOLUME"))
    ]
    if matched:
        region_matches[region] = matched

# --------------------------
# 5. Report results
# --------------------------
print(f"\nMatched {len(region_matches)} regions out of {len(regions)} Freesurfer measures.")
for r, cols in list(region_matches.items())[:15]:
    print(f"{r}: {cols}")

unmatched = [r for r in regions["Measure"].str.upper() if r not in region_matches]
if unmatched:
    print(f"\nUnmatched regions ({len(unmatched)}): {unmatched}")

# --------------------------
# 6. Save mapping
# --------------------------
mapping_df = pd.DataFrame(
    [(r, ", ".join(cols)) for r, cols in region_matches.items()],
    columns=["RegionMeasure", "Columns"]
)

# Merge with regions to include 'Region' column
regions_upper = regions.copy()
regions_upper["RegionMeasure"] = regions_upper["Measure"].str.upper()
mapping_df = mapping_df.merge(regions_upper[["RegionMeasure", "Region"]], on="RegionMeasure", how="left")

# Reorder columns
mapping_df = mapping_df[["RegionMeasure", "Region", "Columns"]]

mapping_df.to_csv(args.output, index=False)
print(f"\nSaved mapping with region info → {args.output}")

# --------------------------
# 7. Check unmatched columns from data (SUVR and VOLUME)
# --------------------------
matched_columns = set([col for cols in region_matches.values() for col in cols])
suvr_vol_cols = [col for col in data_cols if col.endswith("_SUVR") or col.endswith("_VOLUME")]
unmatched_data_cols = sorted(set(suvr_vol_cols) - matched_columns)

print(f"\nFound {len(unmatched_data_cols)} SUVR/VOLUME columns not mapped to any Freesurfer region.")
if len(unmatched_data_cols) > 0:
    print("Unmatched columns:")
    print(unmatched_data_cols)

# Save unmatched columns to a file for reference
unmatched_path = args.output.replace(".csv", "_unmatched_columns.csv")
pd.DataFrame(unmatched_data_cols, columns=["UnmatchedColumns"]).to_csv(unmatched_path, index=False)
print(f"Saved unmatched columns → {unmatched_path}")