#!/usr/bin/env python3
"""
Convert a subject's gate_weights from per-subject JSON into ggseg-compatible CSV
for brain_plot_clean_zixuan.R (columns: ggseg_dk, region, mri, amy, dk).

Usage (example):
  python3 scripts/subject_gate_to_brain_csv.py \
    --input results/moe_seed_7_full_final_per_subject.json \
    --ptid 021_S_4558 \
        --output results/subject_CN_021_S_4558_brain.csv

Usage with plotting (example):
    python3 scripts/subject_gate_to_brain_csv.py \
        --input results/moe_seed_1234_full_final_per_subject.json \
        --ptid 021_S_4558 \
        --output results/subject_CN_021_S_4558_brain.csv \
        --experts_yaml configs/experts.yaml \
        --columns_csv data/UCBERKELEY_AMY_6MM_08Jul2024.csv

    python3 scripts/subject_gate_to_brain_csv.py \
        --input results/moe_seed_1234_full_final_per_subject.json \
        --ptid 068_S_2171 \
        --output results/subject_MCI_068_S_2171_brain.csv \
        --experts_yaml configs/experts.yaml \
        --columns_csv data/UCBERKELEY_AMY_6MM_08Jul2024.csv
    
    python3 scripts/subject_gate_to_brain_csv.py \
        --input results/moe_seed_1234_full_final_per_subject.json \
        --ptid 031_S_4203 \
        --output results/subject_AD_031_S_4203_brain.csv \
        --experts_yaml configs/experts.yaml \
        --columns_csv data/UCBERKELEY_AMY_6MM_08Jul2024.csv
    

Notes:
- If --ptid is not provided, you can use --index to select a subject by position.
- The output CSV is compatible with scripts/brain_plot_clean_zixuan.R
- Optional: generate a cortical surface plot directly in Python (requires nilearn).
"""

import argparse
import json
import os
import re
from typing import Dict, Any

import pandas as pd
import yaml


def load_subjects(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of subjects in the JSON file.")
    return data


def pick_subject(subjects, ptid=None, index=None):
    if ptid is not None:
        for s in subjects:
            if s.get("PTID") == ptid:
                return s
        raise ValueError(f"PTID not found: {ptid}")
    if index is None:
        raise ValueError("Provide either --ptid or --index.")
    if index < 0 or index >= len(subjects):
        raise IndexError(f"Index out of range: {index}")
    return subjects[index]


def is_subcortical(region: str) -> bool:
    r = region.lower()
    return any(x in r for x in [
        "subcortical",
        "ventricle",
        "brainstem",
        "cerebellum",
        "corpus",
    ])


def to_ggseg_label(region: str) -> str:
    return region.replace("_", " ").lower()


# Measure (abbreviation) → ggseg DK label mapping from freesurfer_brain_regions.csv
# NOTE: Only includes individual brain regions that ggseg can display, not summary measures
FREESURFER_TO_GGSEG = {
    "AmygVol": "amygdala",
    "HippVol": "hippocampus",
    "AccumVol": "accumbens area",
    "CaudVol": "caudate",
    "PallVol": "pallidum",
    "PutamVol": "putamen",
    "ThalVol": "thalamus proper",
    "CerebellCtx": "cerebellum cortex",
    "CerebellWM": "cerebellum white matter",
    "InfLatVent": "inferior lateral ventricle",
    "LatVent": "lateral ventricle",
    "CSF": "csf",
    "CC_Post": "CC posterior",
    "CC_MidPost": "CC mid posterior",
    "CC_Cent": "CC central",
    "CC_MidAnt": "CC mid anterior",
    "CC_Ant": "CC anterior",
    "BrainStem": "brain stem",
    "CaudMidFrontal": "caudal middle frontal",
    "FrontalPole": "frontal pole",
    "LatOrbFrontal": "lateral orbitofrontal",
    "MedOrbFrontal": "medial orbitofrontal",
    "ParsOper": "pars opercularis",
    "ParsOrb": "pars orbitalis",
    "ParsTriang": "pars triangularis",
    "RostMidFrontal": "rostral middle frontal",
    "SupFrontal": "superior frontal",
    "CaudAntCing": "caudal anterior cingulate",
    "IsthmCing": "isthmus cingulate",
    "PostCing": "posterior cingulate",
    "RostAntCing": "rostral anterior cingulate",
    "InfParietal": "inferior parietal",
    "Precuneus": "precuneus",
    "SupParietal": "superior parietal",
    "Supramarg": "supramarginal",
    "BanksSTS": "bankssts",
    "EntCtx": "entorhinal",
    "Fusiform": "fusiform",
    "InfTemporal": "inferior temporal",
    "Lingual": "lingual",
    "MidTemporal": "middle temporal",
    "Parahipp": "parahippocampal",
    "SupTemporal": "superior temporal",
    "TemporalPole": "temporal pole",
    "TransvTemporal": "transverse temporal",
    "Cuneus": "cuneus",
    "LatOccipital": "lateral occipital",
    "Pericalc": "pericalcarine",
    "Paracentral": "paracentral",
    "Postcentral": "postcentral",
    "Precentral": "precentral",
    # Additional column name mappings from actual data CSV
    "AMYGDALA": "amygdala",
    "HIPPOCAMPUS": "hippocampus",
    "ACCUMBENS_AREA": "accumbens area",
    "CAUDATE": "caudate",
    "PALLIDUM": "pallidum",
    "PUTAMEN": "putamen",
    "THALAMUS_PROPER": "thalamus proper",
    "VENTRALDC": "ventral DC",
    "BRAINSTEM": "brain stem",
    "CEREBELLUM_CORTEX": "cerebellum cortex",
    "CEREBELLUM_WHITE_MATTER": "cerebellum white matter",
    "CEREBRAL_WHITE_MATTER": "cerebral white matter",
    "INF_LAT_VENT": "inferior lateral ventricle",
    "LATERAL_VENTRICLE": "lateral ventricle",
    "VENTRICLE_3RD": "3rd ventricle",
    "VENTRICLE_4TH": "4th ventricle",
    "VENTRICLE_5TH": "5th ventricle",
    "CSF": "csf",
    "CHOROID_PLEXUS": "choroid plexus",
    "OPTIC_CHIASM": "optic chiasm",
    "VESSEL": "vessel",
    "CC_ANTERIOR": "CC anterior",
    "CC_CENTRAL": "CC central",
    "CC_MID_ANTERIOR": "CC mid anterior",
    "CC_MID_POSTERIOR": "CC mid posterior",
    "CC_POSTERIOR": "CC posterior",
    "BANKSSTS": "bankssts",
    "CAUDALANTERIORCINGULATE": "caudal anterior cingulate",
    "CAUDALMIDDLEFRONTAL": "caudal middle frontal",
    "CUNEUS": "cuneus",
    "ENTORHINAL": "entorhinal",
    "FRONTALPOLE": "frontal pole",
    "FUSIFORM": "fusiform",
    "INFERIORPARIETAL": "inferior parietal",
    "INFERIORTEMPORAL": "inferior temporal",
    "INSULA": "insula",
    "ISTHMUSCINGULATE": "isthmus cingulate",
    "LATERALOCCIPITAL": "lateral occipital",
    "LATERALORBITOFRONTAL": "lateral orbitofrontal",
    "LINGUAL": "lingual",
    "MEDIALORBITOFRONTAL": "medial orbitofrontal",
    "MIDDLETEMPORAL": "middle temporal",
    "PARACENTRAL": "paracentral",
    "PARAHIPPOCAMPAL": "parahippocampal",
    "PARSOPERCULARIS": "pars opercularis",
    "PARSORBITALIS": "pars orbitalis",
    "PARSTRIANGULARIS": "pars triangularis",
    "PERICALCARINE": "pericalcarine",
    "POSTCENTRAL": "postcentral",
    "POSTERIORCINGULATE": "posterior cingulate",
    "PRECENTRAL": "precentral",
    "PRECUNEUS": "precuneus",
    "ROSTRALANTERIORCINGULATE": "rostral anterior cingulate",
    "ROSTRALMIDDLEFRONTAL": "rostral middle frontal",
    "SUPERIORFRONTAL": "superior frontal",
    "SUPERIORPARIETAL": "superior parietal",
    "SUPERIORTEMPORAL": "superior temporal",
    "SUPRAMARGINAL": "supramarginal",
    "TEMPORALPOLE": "temporal pole",
    "TRANSVERSETEMPORAL": "transverse temporal",
}

GGSEG_REGION_LABELS = list(set(FREESURFER_TO_GGSEG.values()))


def _normalize_ggseg_key(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


GGSEG_REGION_MAP = {
    _normalize_ggseg_key(label): label for label in GGSEG_REGION_LABELS
}

GGSEG_ALIAS_MAP = {
    "brainstem": "brain stem",
    "lateralventricle": "lateral ventricle",
    "inferiorlateralventricle": "inferior lateral ventricle",
    "thalamus": "thalamus proper",
    "ventraldc": "ventral DC",
    "ccposterior": "CC posterior",
    "ccmidposterior": "CC mid posterior",
    "cccentral": "CC central",
    "ccmidanterior": "CC mid anterior",
    "ccanterior": "CC anterior",
}


MAPPING_PATTERNS = {
    "AMYGVOL": ["AMYGDALA"],
    "HIPPVOL": ["HIPPOCAMPUS"],
    "ACCUMVOL": ["ACCUMBENS_AREA"],
    "CAUDVOL": ["CAUDATE"],
    "PALLVOL": ["PALLIDUM"],
    "PUTAMVOL": ["PUTAMEN"],
    "THALVOL": ["THALAMUS"],
    "CEREBCTX": ["CEREBRAL_CORTEX"],
    "CEREBWM": ["CEREBRAL_WHITE_MATTER"],
    "CEREBELLCTX": ["CEREBELLUM_CORTEX"],
    "CEREBELLWM": ["CEREBELLUM_WHITE_MATTER"],
    "INFLATVENT": ["INFERIOR_LATERAL_VENTRICLE"],
    "LATVENT": ["LATERAL_VENTRICLE"],
    "CSF": ["CSF"],
    "CC_POST": ["CC_POSTERIOR"],
    "CC_MIDPOST": ["CC_MID_POSTERIOR"],
    "CC_CENT": ["CC_CENTRAL"],
    "CC_MIDANT": ["CC_MID_ANTERIOR"],
    "CC_ANT": ["CC_ANTERIOR"],
    "BRAINSTEM": ["BRAINSTEM"],
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


def _normalize_region_key(text: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _compile_patterns(items):
    return [re.compile(p) for p in (items or [])]


def _match_any(name: str, pats) -> bool:
    return any(p.search(name) for p in pats)


def _select_by_regex(columns, include_regex, exclude_regex):
    inc = _compile_patterns(include_regex)
    exc = _compile_patterns(exclude_regex)
    out = []
    for c in columns:
        if inc and not _match_any(c, inc):
            continue
        if exc and _match_any(c, exc):
            continue
        out.append(c)
    return out


def _column_to_ggseg_label(col: str) -> str:
    """Convert FreeSurfer data column name (like AMYGDALA_SUVR or CTX_ENTORHINAL_VOLUME) to ggseg label.
    
    Returns None if the column doesn't map to a valid ggseg region.
    """
    name = col.upper()
    
    # Remove measurement suffix (_SUVR, _VOLUME, etc.)
    for suf in ["_SUVR", "_VOLUME", "_VOLUMN"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    
    # Remove hemisphere prefix (CTX_LH_, CTX_RH_, LEFT_, RIGHT_, CTX_)
    for pref in ["CTX_LH_", "CTX_RH_", "LEFT_", "RIGHT_", "CTX_"]:
        if name.startswith(pref):
            name = name[len(pref):]
            break
    
    # Try exact match in FREESURFER_TO_GGSEG (most specific)
    if name in FREESURFER_TO_GGSEG:
        return FREESURFER_TO_GGSEG[name]
    
    # Try normalized key lookup in GGSEG_REGION_MAP
    key = _normalize_ggseg_key(name)
    if key in GGSEG_ALIAS_MAP:
        return GGSEG_ALIAS_MAP[key]
    if key in GGSEG_REGION_MAP:
        return GGSEG_REGION_MAP[key]
    
    # If not found in any mapping, return None to filter it out
    return None


def _measure_to_ggseg_name(measure: str) -> str | None:
    """Convert FreeSurfer measure name to ggseg label, or None if not a valid ggseg region."""
    # Try exact match first
    if measure in FREESURFER_TO_GGSEG:
        return FREESURFER_TO_GGSEG[measure]
    
    # Try case-insensitive match
    for key, value in FREESURFER_TO_GGSEG.items():
        if key.lower() == measure.lower():
            return value
    
    # Return None for unmapped measures
    return None


def build_region_mapping(mapping_csv_path: str) -> Dict[str, list]:
    """Build mapping from coarse region names (e.g., 'temporal_lobe') to fine-grained ggseg labels.
    
    Uses freesurfer_brain_regions.csv to expand coarse region categories into their
    constituent measures, then converts those to ggseg labels.
    """
    if not os.path.isfile(mapping_csv_path):
        return {}
    regions_df = pd.read_csv(mapping_csv_path)
    mapping: Dict[str, list] = {}
    
    for _, row in regions_df.iterrows():
        region_group = str(row.get("Region", "")).strip()
        measure = str(row.get("Measure", "")).strip()
        if not region_group or not measure:
            continue
        
        # Normalize the region group name (e.g., "Temporal Lobe" -> "temporal_lobe")
        group_key = _normalize_region_key(region_group)
        
        # Convert measure to ggseg label
        ggseg_label = _measure_to_ggseg_name(measure)
        
        # Only add valid ggseg labels
        if ggseg_label is not None:
            mapping.setdefault(group_key, []).append(ggseg_label)
    
    # Remove duplicates within each group
    for key in mapping:
        mapping[key] = sorted(set(mapping[key]))
    
    return mapping


def build_region_mapping_from_experts(experts_yaml: str, columns_csv: str) -> Dict[str, list]:
    """Build region mapping from experts.yaml groups applied to column names.
    
    Maps from gate weight keys (e.g., 'temporal_lobe') to individual ggseg regions
    (e.g., ['amygdala', 'hippocampus', 'entorhinal', 'fusiform', ...])
    """
    if not os.path.isfile(experts_yaml) or not os.path.isfile(columns_csv):
        return {}
    with open(experts_yaml, "r") as f:
        cfg = yaml.safe_load(f) or {}
    groups = cfg.get("groups", {})
    if not isinstance(groups, dict):
        return {}

    columns = list(pd.read_csv(columns_csv, nrows=0).columns)
    mapping: Dict[str, list] = {}

    for group_name, spec in groups.items():
        if "demographic" in group_name.lower() or "summary" in group_name.lower():
            continue
        include_regex = (spec or {}).get("include_regex", [])
        exclude_regex = (spec or {}).get("exclude_regex", [])
        matched = _select_by_regex(columns, include_regex, exclude_regex)
        if not matched:
            continue
        
        # Extract region key from group name (e.g., 'amy_temporal_lobe' -> 'temporal_lobe')
        region_key = group_name
        if region_key.startswith("amy_"):
            region_key = region_key[len("amy_"):]
        if region_key.startswith("mri_"):
            region_key = region_key[len("mri_"):]
        region_key = region_key.lower()
        
        # Convert each matched column to ggseg label
        labels = []
        for col in matched:
            ggseg_label = _column_to_ggseg_label(col)
            if ggseg_label is not None:  # Filter out None values
                labels.append(ggseg_label)
        
        # Remove duplicates and sort
        labels = sorted(set(labels))
        if labels:
            mapping[region_key] = labels
    
    return mapping


def build_rows(gate_weights: Dict[str, float], region_mapping: Dict[str, list] | None = None, fallback_mapping: Dict[str, list] | None = None):
    """Build rows for CSV output from gate weights.
    
    Attempts to map each gate weight key to ggseg regions:
    1. First tries region_mapping (from experts.yaml + column regex matching)
    2. Then tries fallback_mapping (from coarse region categories in freesurfer_brain_regions.csv)
    3. Finally tries direct label conversion as last resort
    """
    rows = {}

    for key, value in gate_weights.items():
        if key == "demographic" or "demo" in key.lower():
            continue

        if key.startswith("mri_"):
            modality = "mri"
            region = key[len("mri_"):]
        elif key.startswith("amy_"):
            modality = "amy"
            region = key[len("amy_"):]
        else:
            continue

        mapped_labels = None
        
        # Try primary mapping (from experts.yaml)
        if region_mapping is not None:
            mapped_labels = region_mapping.get(region)
        
        # If no mapping found, try fallback mapping (from coarse region categories)
        if not mapped_labels and fallback_mapping is not None:
            mapped_labels = fallback_mapping.get(region)

        if mapped_labels:
            # Only include labels that are actually in the ggseg atlas
            for ggseg_label in mapped_labels:
                # Verify the label is in the ggseg region map (normalized)
                normalized_label = _normalize_ggseg_key(ggseg_label)
                if normalized_label not in GGSEG_REGION_MAP and ggseg_label not in GGSEG_REGION_LABELS:
                    continue  # Skip labels not in ggseg
                
                if ggseg_label not in rows:
                    rows[ggseg_label] = {
                        "ggseg_dk": ggseg_label,
                        "region": ggseg_label,
                        "mri": 0.0,
                        "amy": 0.0,
                        "dk": 0 if is_subcortical(region) else 1,
                    }
                rows[ggseg_label][modality] = float(value)
        else:
            # Last resort fallback: try to convert region name to ggseg label
            region_normalized = _normalize_ggseg_key(region)
            
            # Check alias map first
            if region_normalized in GGSEG_ALIAS_MAP:
                ggseg_label = GGSEG_ALIAS_MAP[region_normalized]
            # Check main region map
            elif region_normalized in GGSEG_REGION_MAP:
                ggseg_label = GGSEG_REGION_MAP[region_normalized]
            # Check FREESURFER_TO_GGSEG
            elif region.upper() in FREESURFER_TO_GGSEG:
                ggseg_label = FREESURFER_TO_GGSEG[region.upper()]
            else:
                continue  # Skip unmappable regions
            
            # Validate that the label is in ggseg regions
            normalized_label = _normalize_ggseg_key(ggseg_label)
            if normalized_label not in GGSEG_REGION_MAP and ggseg_label not in GGSEG_REGION_LABELS:
                continue  # Skip labels not in ggseg
            
            if ggseg_label not in rows:
                rows[ggseg_label] = {
                    "ggseg_dk": ggseg_label,
                    "region": ggseg_label,
                    "mri": 0.0,
                    "amy": 0.0,
                    "dk": 0 if is_subcortical(region) else 1,
                }
            rows[ggseg_label][modality] = float(value)

    return list(rows.values())


def _normalize_label(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def plot_cortical_surface(df: pd.DataFrame, modality: str, output_path: str):
    """Plot cortical surface using nilearn's Desikan-Killiany atlas.

    Note: This only visualizes cortical regions (dk == 1). Subcortical regions
    are not shown in surface plots.
    """
    try:
        import numpy as np
        from nilearn import datasets, plotting
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "nilearn is required for plotting. Install with: pip install nilearn"
        ) from exc

    if modality not in {"mri", "amy"}:
        raise ValueError("modality must be 'mri' or 'amy'")

    cortical = df[df["dk"] == 1].copy()
    if cortical.empty:
        raise ValueError("No cortical regions found to plot.")

    # Build mapping from region label to value
    region_to_val = {
        _normalize_label(r): float(v)
        for r, v in zip(cortical["ggseg_dk"], cortical[modality])
    }

    if hasattr(datasets, "fetch_atlas_surf_desikan_killiany"):
        atlas = datasets.fetch_atlas_surf_desikan_killiany()
    else:
        raise SystemExit(
            "This script requires the FreeSurfer Desikan-Killiany surface atlas. "
            "Please upgrade nilearn (e.g., pip install -U nilearn) or use the R plotting script."
        )
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")

    labels = atlas["labels"]
    map_left = atlas["map_left"]
    map_right = atlas["map_right"]

    # Map atlas labels to region weights
    label_to_val = {}
    for idx, label in enumerate(labels):
        if label is None:
            continue
        norm = _normalize_label(label)
        if norm in region_to_val:
            label_to_val[idx] = region_to_val[norm]

    # Build per-vertex data
    left_vals = np.zeros_like(map_left, dtype=float)
    right_vals = np.zeros_like(map_right, dtype=float)
    for idx, val in label_to_val.items():
        left_vals[map_left == idx] = val
        right_vals[map_right == idx] = val

    vmin = min(left_vals.min(), right_vals.min())
    vmax = max(left_vals.max(), right_vals.max())

    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_left"],
        left_vals,
        hemi="left",
        view="lateral",
        colorbar=True,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        title=f"{modality.upper()} cortical gate weights (left-lateral)",
    )
    fig.savefig(output_path.replace(".png", "_left_lateral.png"), dpi=300, bbox_inches="tight")
    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_left"],
        left_vals,
        hemi="left",
        view="medial",
        colorbar=True,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        title=f"{modality.upper()} cortical gate weights (left-medial)",
    )
    fig.savefig(output_path.replace(".png", "_left_medial.png"), dpi=300, bbox_inches="tight")
    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_right"],
        right_vals,
        hemi="right",
        view="lateral",
        colorbar=True,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        title=f"{modality.upper()} cortical gate weights (right-lateral)",
    )
    fig.savefig(output_path.replace(".png", "_right_lateral.png"), dpi=300, bbox_inches="tight")
    fig = plotting.plot_surf_stat_map(
        fsaverage["infl_right"],
        right_vals,
        hemi="right",
        view="medial",
        colorbar=True,
        cmap="mako_r",
        vmin=vmin,
        vmax=vmax,
        title=f"{modality.upper()} cortical gate weights (right-medial)",
    )
    fig.savefig(output_path.replace(".png", "_right_medial.png"), dpi=300, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Per-subject JSON file")
    ap.add_argument("--ptid", default=None, help="Subject PTID to select")
    ap.add_argument("--index", type=int, default=None, help="Subject index to select")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--mapping_csv", default="data/freesurfer_brain_regions.csv", help="CSV mapping for FreeSurfer regions")
    ap.add_argument("--experts_yaml", default=None, help="Experts YAML with regex groups (mesoscale → FreeSurfer columns)")
    ap.add_argument("--columns_csv", default=None, help="CSV file to supply FreeSurfer column names for regex matching")
    ap.add_argument("--plot", action="store_true", help="Generate cortical surface plot (nilearn required)")
    ap.add_argument("--plot_modality", choices=["mri", "amy"], default="mri", help="Modality to plot")
    ap.add_argument("--plot_output", default=None, help="Output PNG path prefix for plots")
    args = ap.parse_args()

    subjects = load_subjects(args.input)
    subject = pick_subject(subjects, ptid=args.ptid, index=args.index)

    gate_weights = subject.get("gate_weights")
    if not gate_weights:
        raise ValueError("Selected subject has no gate_weights.")

    # Load primary region mapping (from experts.yaml)
    region_mapping = {}
    if args.experts_yaml and args.columns_csv:
        region_mapping = build_region_mapping_from_experts(args.experts_yaml, args.columns_csv)
    
    # Load fallback mapping (from coarse region categories)
    fallback_mapping = build_region_mapping(args.mapping_csv)
    
    rows = build_rows(gate_weights, region_mapping, fallback_mapping)
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    ptid = subject.get("PTID", "unknown")
    print(f"Wrote CSV for PTID={ptid} → {args.output}")

    if args.plot:
        if args.plot_output is None:
            base = os.path.splitext(args.output)[0]
            plot_output = f"{base}_{args.plot_modality}.png"
        else:
            plot_output = args.plot_output
        plot_cortical_surface(df, args.plot_modality, plot_output)
        print(f"Saved cortical surface plots with prefix → {plot_output}")


if __name__ == "__main__":
    main()
