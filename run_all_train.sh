#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CONFIG="config/Config.yaml"

run_train() {
    dataset="$1"
    network="$2"
    rate="$3"
    momentum="$4"
    smoothing="$5"

    python3 - "$CONFIG" "$dataset" "$network" "$rate" "$momentum" "$smoothing" <<'PY'
import sys, yaml

path, dataset, network, rate, momentum, smoothing = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f)

cfg["Dataset"] = dataset
cfg["Network"] = network
cfg["R"] = int(rate)
cfg["Momentum_type"] = momentum
cfg["Smoothing_type"] = smoothing

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "Training: $network | $dataset | R=$rate | $momentum | $smoothing"
    python3 -u train.py
}

# PDDL: CorPD and CorPDFS at R4, R6, and R8
for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_train "$dataset" PDDL "$rate" None Simple
    done
done

# UMPIRE: CorPD and CorPDFS at R4, R6, and R8
# Each is trained with Nesterov and without momentum.
for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_train "$dataset" UMPIRE "$rate" Nesterov Simple
        run_train "$dataset" UMPIRE "$rate" None Simple
    done
done

# Additional CorPD R8 ablations
run_train CorPD UMPIRE 8 None LogExp
run_train CorPD UMPIRE 8 None SmoothL1
run_train CorPD UMPIRE 8 Polyak Simple
