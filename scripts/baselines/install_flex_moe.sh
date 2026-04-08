#!/usr/bin/env bash
set -euo pipefail

# Install Flex-MoE and (optionally) FastMoE from source.
# Usage:
#   bash scripts/install_flex_moe.sh
#   bash scripts/install_flex_moe.sh --skip-fastmoe
#   FLEX_MOE_DIR=third_party/flex-moe bash scripts/install_flex_moe.sh

SKIP_FASTMOE=false
ALLOW_BREAK_SYSTEM_PACKAGES=false
for arg in "$@"; do
  case "${arg}" in
    --skip-fastmoe) SKIP_FASTMOE=true ;;
    --break-system-packages) ALLOW_BREAK_SYSTEM_PACKAGES=true ;;
    *)
      echo "[ERROR] Unknown argument: ${arg}"
      echo "Usage: bash scripts/install_flex_moe.sh [--skip-fastmoe] [--break-system-packages]"
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
FLEX_MOE_DIR="${FLEX_MOE_DIR:-${THIRD_PARTY_DIR}/flex-moe}"
FASTMOE_DIR="${FASTMOE_DIR:-${THIRD_PARTY_DIR}/fastmoe}"

mkdir -p "${THIRD_PARTY_DIR}"

echo "[INFO] Repo root: ${ROOT_DIR}"
echo "[INFO] Flex-MoE dir: ${FLEX_MOE_DIR}"
echo "[INFO] FastMoE dir: ${FASTMOE_DIR}"

if [[ -z "${CONDA_PREFIX:-}" ]] && [[ -z "${VIRTUAL_ENV:-}" ]] && [[ "${ALLOW_BREAK_SYSTEM_PACKAGES}" != "true" ]]; then
  echo "[ERROR] No active conda/venv detected."
  echo "Activate your project environment first (recommended), e.g.:"
  echo "  conda activate ad-moe"
  echo "Or rerun with --break-system-packages (not recommended)."
  exit 1
fi

PIP_FLAGS=()
if [[ "${ALLOW_BREAK_SYSTEM_PACKAGES}" == "true" ]]; then
  PIP_FLAGS+=(--break-system-packages)
fi

if [[ ! -d "${FLEX_MOE_DIR}/.git" ]]; then
  echo "[INFO] Cloning Flex-MoE..."
  git clone https://github.com/UNITES-Lab/flex-moe.git "${FLEX_MOE_DIR}"
else
  echo "[INFO] Flex-MoE already exists. Pulling latest main..."
  git -C "${FLEX_MOE_DIR}" pull --ff-only
fi

echo "[INFO] Installing Python dependencies used by Flex-MoE..."
python -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel
if [[ -f "${FLEX_MOE_DIR}/requirements.txt" ]]; then
  python -m pip install "${PIP_FLAGS[@]}" -r "${FLEX_MOE_DIR}/requirements.txt"
else
  echo "[INFO] No requirements.txt found in Flex-MoE; installing known dependencies."
  python -m pip install "${PIP_FLAGS[@]}" dm-tree scikit-learn tqdm pandas scanpy nibabel
fi

if [[ "${SKIP_FASTMOE}" == "true" ]]; then
  echo "[INFO] --skip-fastmoe set; skipping FastMoE build."
else
  if [[ ! -d "${FASTMOE_DIR}/.git" ]]; then
    echo "[INFO] Cloning FastMoE..."
    git clone https://github.com/laekov/fastmoe.git "${FASTMOE_DIR}"
  else
    echo "[INFO] FastMoE already exists. Pulling latest main..."
    git -C "${FASTMOE_DIR}" pull --ff-only
  fi

  echo "[INFO] Building FastMoE (USE_NCCL=0)..."
  USE_NCCL=0 python -m pip install "${PIP_FLAGS[@]}" -e "${FASTMOE_DIR}"
fi

echo "[INFO] Flex-MoE setup complete."
echo "[INFO] Next step: adapt Flex-MoE data loader to this repo's experts/splits format."
