#!/usr/bin/env bash
set -Eeuo pipefail

# Run from the UMPIRE-Net repository root, where Config.yaml and train.py live.
# Usage: ./run_train.sh [cuda:0]
# Set FORCE=1 to retrain experiments that already contain a checkpoint.

DEVICE="${1:-cuda:0}"
CONFIG="Config.yaml"
PYTHON="${PYTHON:-python3}"
FORCE="${FORCE:-0}"
LOG_DIR="logs/train"

[[ -f "$CONFIG" ]] || { echo "Missing $CONFIG" >&2; exit 1; }
[[ -f train.py ]] || { echo "Missing train.py" >&2; exit 1; }
mkdir -p "$LOG_DIR"

CONFIG_BACKUP="$(mktemp ./Config.yaml.backup.XXXXXX)"
cp "$CONFIG" "$CONFIG_BACKUP"
restore_config() { cp "$CONFIG_BACKUP" "$CONFIG"; rm -f "$CONFIG_BACKUP"; }
trap restore_config EXIT INT TERM

set_config() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5"
    "$PYTHON" - "$CONFIG" "$DEVICE" "$dataset" "$network" "$rate" "$momentum" "$smoothing" <<'PY'
import sys, yaml

path, device, dataset, network, rate, momentum, smoothing = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f)

cfg.update({
    "device": device,
    "Dataset": dataset,
    "Network": network,
    "R": int(rate),
    "Momentum_type": momentum,
    "Smoothing_type": smoothing,
    "Share_params": False,
    "SoftPlus": True,
    "GD_or_CG": "CG",
    "PF_Factor": 0.75,
    "epsilon": 1e-6,
})
with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

experiment_name() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5"
    if [[ "$network" == "PDDL" ]]; then
        printf 'PDDL_LARGE_%s_R%s_UNet_Share0_CG_SoftPlus_0.75' "$dataset" "$rate"
    else
        printf 'UMPIRE_%s_R%s_UNet_Share0_SoftPlus_%s_0.75_epsilon1e-06_Smoothing_%s' \
            "$dataset" "$rate" "$momentum" "$smoothing"
    fi
}

run_one() {
    local dataset="$1" network="$2" rate="$3" momentum="$4" smoothing="$5"
    local name checkpoint_glob
    name="$(experiment_name "$dataset" "$network" "$rate" "$momentum" "$smoothing")"
    checkpoint_glob="Train_Results/$name/model/BestModel_Val_R${rate}_Epoch"'*.pth'

    if [[ "$FORCE" != "1" ]] && compgen -G "$checkpoint_glob" >/dev/null; then
        echo "[skip] $name (checkpoint exists)"
        return
    fi

    echo "[train] $name on $DEVICE"
    set_config "$dataset" "$network" "$rate" "$momentum" "$smoothing"
    "$PYTHON" -u train.py 2>&1 | tee "$LOG_DIR/$name.log"
}

# PDDL baselines shown in the experiment list. The uploaded train.py always
# inserts LARGE into every PDDL result name, including CorPD/PDFS R4 runs.
for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_one "$dataset" PDDL "$rate" None Simple
    done
done
for rate in 6 8; do
    run_one AxFLAIR PDDL "$rate" None Simple
done

# Main UMPIRE-Net experiments.
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

# CorPD R8 ablations shown in the experiment list.
run_one CorPD UMPIRE 8 None LogExp
run_one CorPD UMPIRE 8 None SmoothL1
run_one CorPD UMPIRE 8 Polyak Simple

echo "All requested training runs are complete."
