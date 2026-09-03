#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

DATASET_PATH="${1:-data/dataset-kitchen}"
PLY_PATH="${2:-_output/pbr-stage2-full/optimized-2d-gaussians_pbr_arm.ply}"
SPP="${3:-1}"
BOUNCE_LIMIT="${4:-1}"
SCENARIO="${5:-0}"
OUTPUT_SUFFIX="${6:-pbr-p4-spp${SPP}-bounce${BOUNCE_LIMIT}-scenario${SCENARIO}}"

mkdir -p logs
LOG_PATH="logs/${OUTPUT_SUFFIX}.log"

python editing-and-rendering.py \
  --PBR_ENABLED true \
  --DATASET_IS_SYNTHETIC true \
  --NVS_DATASET_PATH "${DATASET_PATH}" \
  --EAG_PLY_PATH "${PLY_PATH}" \
  --I_SCENE_EDITING_SCENARIO "${SCENARIO}" \
  --PBR_PT_SPP "${SPP}" \
  --PBR_PT_BOUNCE_LIMIT "${BOUNCE_LIMIT}" \
  --PBR_PT_USE_DISNEY_SAMPLING true \
  --PBR_PT_USE_NEE true \
  --PBR_PT_USE_MIS true \
  --PBR_PT_USE_RUSSIAN_ROULETTE true \
  --PBR_PT_RR_START_BOUNCE 3 \
  --PBR_PT_LIGHT_SAMPLES 1 \
  --PBR_PT_SAVE_DECOMPOSITION true \
  --PBR_PT_RENDER_NOBOUNCE false \
  --PBR_PT_RENDER_SINGLEBOUNCE false \
  --PBR_PT_RENDER_PATH_TRACING true \
  --OUTPUT_FOLDER_SUFFIX "${OUTPUT_SUFFIX}" \
  2>&1 | tee "${LOG_PATH}"

if grep -Ei '(^|[^[:alpha:]])(nan|inf)([^[:alpha:]]|$)|cuda error|optix.*error|traceback' \
  "${LOG_PATH}"; then
  echo "P4 validation found an error pattern in ${LOG_PATH}" >&2
  exit 1
fi

echo "P4 render completed: _output/${OUTPUT_SUFFIX}"
