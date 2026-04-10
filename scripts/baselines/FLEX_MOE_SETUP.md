# Flex-MoE Setup (for this repo)

This project can use upstream Flex-MoE code from source.

## 1) Install

From repo root:

```bash
bash scripts/baselines/install_flex_moe.sh
```

If FastMoE build is not needed yet:

```bash
bash scripts/baselines/install_flex_moe.sh --skip-fastmoe
```

If you must use system Python (not recommended), use:

```bash
bash scripts/baselines/install_flex_moe.sh --break-system-packages
```

What this does:
- clones `UNITES-Lab/flex-moe` into `third_party/flex-moe`
- installs Flex-MoE Python requirements
- clones and installs `laekov/fastmoe` into `third_party/fastmoe` (unless skipped)

## 2) Verify import

```bash
python - <<'PY'
import sys
sys.path.insert(0, "third_party/flex-moe")
import moe_module
print("Flex-MoE module import ok:", moe_module.__file__)
PY
```

## 3) Important note for this AD pipeline

Flex-MoE upstream expects its own dataset layout and split format.  
To run with this repo's 3-modality expert CSVs and 10 seeds (like the current baselines), we still need a small adapter that maps:
- `configs/*experts_files.yaml` -> Flex-MoE modality tensors
- `configs/splits/splits_by_ptid_80_10_10_seed_*.json` -> train/val/test indices

This setup file only covers installation and import readiness.
