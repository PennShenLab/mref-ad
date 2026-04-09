# Data preparation

This folder holds **exploratory and helper scripts** used before multimodal training and analysis. It does **not** redistribute raw ADNI (or other) data; you must obtain data under your own agreements and place files under `data/` (and set paths) locally.

## Typical order of operations

These steps reflect a reasonable pipeline; adapt names and paths to your cohort.

1. **Raw / multi-visit tables** — Place ADNIMERGE-style or study-specific CSVs under `data/` as needed for your project (e.g. demographics, amyloid/MRI tables).

2. **Understand the cohort (optional)** — One-off exploration scripts:
   - `explore_diagnosis.py`, `explore_demographics.py`, `explore_subjects.py`, `summarize_participants.py` — counts and overlaps.

3. **FreeSurfer region list** — `explore_freesurfer_brain_regions.py` reads the VBM keys spreadsheet and writes `data/freesurfer_brain_regions.csv`.

4. **Map regions to dataset columns** — `map_freesurfer_brain_regions_to_data.py` produces a mapping CSV (e.g. `data/freesurfer_to_column_mapping.csv`) from regions + an imaging table.

5. **Experts YAML (optional automation)** — `generate_freesurfer_experts_yaml.py` can build an experts config from the mapping file. The paper-style setup may instead use a hand-maintained file such as `configs/freesurfer_lastvisit_experts_files.yaml`, which lists **per-expert CSV paths** under `data/freesurfer_lastvisit/` (or your chosen directory).

6. **Train / val / test splits (per RNG seed)** — `make_splits.py` writes JSON with `train_ptids`, `val_ptids`, and `test_ptids`, stratified by label and grouped by `PTID`. Default is **80% / 10% / 10%**. Run once per seed; output names get `_seed_<id>` appended. See the script docstring for flags (`--paths`, `--out`, `--test_size`, `--val_size`, `--seed`, `--last_visit_only`).

   Prerequisites: `configs/paths.yaml` (or similar) with resolved AMY/MRI CSV paths, and the shared **`utils.build_dataset`** import used by `make_splits.py`.

## Running commands

Activate the project venv (see **`REPRODUCIBILITY.md`**: `source .venv/bin/activate`); then **`python`** is the pinned interpreter (e.g. 3.12) with dependencies from **`requirements.txt`**. Run everything **from the repository root** so `data/` and `configs/` resolve.

### Cheat sheet (paths are examples—adjust to your files)

| Script | Command |
|--------|---------|
| `explore_diagnosis.py` | `python data_preprocessing/explore_diagnosis.py` |
| `explore_demographics.py` | `python data_preprocessing/explore_demographics.py --demo data/PTDEMOG_30Sep2025.csv --imaging data/freesurfer_lastvisit/250826_DX_AMYLOID_last_visit.csv` |
| `explore_subjects.py` | `python data_preprocessing/explore_subjects.py` — optional `--amy` / `--tau` (see env vars below) |
| `summarize_participants.py` | `python data_preprocessing/summarize_participants.py` — writes `results/table1_participants_*.{csv,tex}` |
| `explore_freesurfer_brain_regions.py` | `python data_preprocessing/explore_freesurfer_brain_regions.py` — needs `data/FS-VBM-keys_SLRedit.xlsx` or set **`MREF_FS_VBM_XLSX`** |
| `map_freesurfer_brain_regions_to_data.py` | `python data_preprocessing/map_freesurfer_brain_regions_to_data.py --regions data/freesurfer_brain_regions.csv --data data/freesurfer_lastvisit/250826_DX_AMYLOID_last_visit.csv --output data/freesurfer_to_column_mapping.csv` |
| `generate_freesurfer_experts_yaml.py` | `python data_preprocessing/generate_freesurfer_experts_yaml.py --mapping data/freesurfer_to_column_mapping.csv --output configs/freesurfer_experts.yaml` |
| `make_splits.py` | `python data_preprocessing/make_splits.py --paths configs/paths.yaml --out configs/splits_by_ptid_80_10_10.json --seed 7` — requires **`utils.build_dataset`** (see below) |

Each script’s module docstring has more detail and flags (e.g. `make_splits.py`: `--last_visit_only`, `--test_size`, `--val_size`).

## Relation to training

Training and baselines read:

- Expert file lists from **`configs/*.yaml`** (e.g. `freesurfer_lastvisit_experts_files.yaml`).
- Fixed splits from **`configs/splits_by_ptid_80_10_10_seed_*.json`** (or your naming convention).

For environment setup and smoke tests, see **`REPRODUCIBILITY.md`** at the repo root.

## Smoke test (local)

From the repository root, with `.venv` activated (or let the script activate it):

```bash
bash data_preprocessing/smoke_test.sh
```

- **`explore_freesurfer_brain_regions.py`** exits **2** if `data/FS-VBM-keys_SLRedit.xlsx` is missing; that is **expected** when you already maintain `data/freesurfer_brain_regions.csv`.
- **`make_splits.py`** needs **`configs/paths.yaml`** and a shared **`utils`** module exposing **`build_dataset`** (not shipped in this repo until you restore it).

Override CSV paths for table summaries:

- `MREF_DEMO_CSV`, `MREF_IMG_CSV` — `summarize_participants.py`
- `MREF_AMY_CSV`, `MREF_TAU_CSV` — `explore_subjects.py`
