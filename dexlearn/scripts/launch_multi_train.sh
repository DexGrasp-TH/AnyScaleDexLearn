#!/usr/bin/env bash
set -euo pipefail

# Launch multiple Hydra training runs concurrently on different GPUs.
# Edit RUN_SPECS below. Each entry uses the form:
#   "GPU_ID|EXP_NAME|HYDRA_OVERRIDES"
# CUDA_VISIBLE_DEVICES maps the selected physical GPU to cuda:0 inside each process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-output/multi_train_logs}"
mkdir -p "${LOG_DIR}"

BASE_OVERRIDES=(
  task=train
  data=humanMulti
  algo=humanMultiHierar
  test_data=DGNMulti
  device=cuda:0
)

RUN_SPECS=(
  "1|debug25|algo.model.type_objective=ce data.sampling.train_unit=record_uniform"
  "2|debug26|algo.model.type_objective=ce data.sampling.train_unit=object_uniform"
)

PIDS=()

cleanup() {
  if ((${#PIDS[@]} > 0)); then
    echo "Stopping ${#PIDS[@]} training process(es)..." >&2
    kill "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r gpu_id exp_name hydra_overrides <<< "${spec}"
  log_file="${LOG_DIR}/${exp_name}.log"
  echo "Launching ${exp_name} on GPU ${gpu_id}; log: ${log_file}"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" dexlearn/main.py \
    "${BASE_OVERRIDES[@]}" \
    exp_name="${exp_name}" \
    ${hydra_overrides} \
    > "${log_file}" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if ((failed)); then
  echo "At least one training run failed. Check ${LOG_DIR}." >&2
  exit 1
fi

echo "All training runs finished successfully."
