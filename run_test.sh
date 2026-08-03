#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the UMPIRE-Net repository root, after training.
# Usage: ./run_test.sh [cuda:0]
# For every experiment, the numerically latest best-validation checkpoint is used.

DEVICE="${1:-cuda:0}"
CONFIG="Config.yaml"
PYTHON="${PYTHON:-python3}"
LOG_DIR="logs/test"

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG" >&2; exit 1; }
[[ -f test.py ]] || { echo "Missing test.py" >&2; exit 1; }
mkdir -p "$LOG_DIR"

CONFIG_BACKUP="$(mktemp ./Config.yaml.backup.XXXXXX)"
cp "$CONFIG" "$CONFIG_BACKUP"
restore_config() { cp "$CONFIG_BACKUP" "$CONFIG"; rm -f "$CONFIG_BACKUP"; }
trap restore_config EXIT INT TERM

experiment_name() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5"
    if [[ "$network" == "PDDL" ]]; then
        printf 'PDDL_LARGE_%s_R%s_UNet_Share0_CG_SoftPlus_0.75' "$dataset" "$rate"
    else
        printf 'UMPIRE_%s_R%s_UNet_Share0_SoftPlus_%s_0.75_epsilon1e-06_Smoothing_%s' \
            "$dataset" "$rate" "$momentum" "$smoothing"
    fi
}

latest_epoch() {
    local name="$1" rate="$2" file epoch latest=-1
    shopt -s nullglob
    local files=("Train_Results/$name/model/BestModel_Val_R${rate}_Epoch"*.pth)
    shopt -u nullglob
    for file in "${files[@]}"; do
        epoch="${file##*Epoch}"
        epoch="${epoch%.pth}"
        if [[ "$epoch" =~ ^[0-9]+$ ]] && (( epoch > latest )); then
            latest="$epoch"
        fi
    done
    (( latest >= 0 )) && printf '%s' "$latest"
}

set_config() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5" epoch="$6"
    "$PYTHON" - "$CONFIG" "$DEVICE" "$dataset" "$network" "$rate" "$momentum" "$smoothing" "$epoch" <<'PY'
import sys, yaml

path, device, dataset, network, rate, momentum, smoothing, epoch = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f)

# Architecture-defining values are top-level in test.py.
cfg.update({"epsilon": 1e-6, "SoftPlus": True})
cfg.setdefault("Test", {}).update({
    "device": device,
    "Dataset": dataset,
    "Network": network,
    "model_epoch": int(epoch),
    "R": int(rate),
    "Momentum_type": momentum,
    "Smoothing_type": smoothing,
    "Share_params": False,
    "PF_Factor": 0.75,
})
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_one() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5"
    local name epoch
    name="$(experiment_name "$dataset" "$network" "$rate" "$momentum" "$smoothing")"
    epoch="$(latest_epoch "$name" "$rate")"

    if [[ -z "$epoch" ]]; then
        echo "[skip] $name (no matching checkpoint)" >&2
        return
    fi

    echo "[test] $name, epoch $epoch, on $DEVICE"
    set_config "$dataset" "$network" "$rate" "$momentum" "$smoothing" "$epoch"
    "$PYTHON" -u test.py 2>&1 | tee "$LOG_DIR/${name}_epoch${epoch}.log"
}

for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_one "$dataset" PDDL "$rate" None Simple
    done
done
for rate in 6 8; do
    run_one AxFLAIR PDDL "$rate" None Simple
done

for dataset in AxFLAIR CorPD CorPDFS; do
    rates=(6 8)
    if [[ "$dataset" != "AxFLAIR" ]]; then
        rates=(4 6 8)
    fi
    for rate in "${rates[@]}"; do
        run_one "$dataset" UMPIRE "$rate" Nesterov Simple
        if [[ "$dataset" != "AxFLAIR" ]]; then
            run_one "$dataset" UMPIRE "$rate" None Simple
        fi
    done
done

run_one CorPD UMPIRE 8 None LogExp
run_one CorPD UMPIRE 8 None SmoothL1
run_one CorPD UMPIRE 8 Polyak Simple

echo "All available checkpoints have been tested."
