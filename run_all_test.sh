#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CONFIG="config/Config.yaml"

experiment_name() {
    dataset="$1"
    network="$2"
    rate="$3"
    momentum="$4"
    smoothing="$5"

    if [ "$network" = "PDDL" ]; then
        echo "PDDL_LARGE_${dataset}_R${rate}_UNet_Share0_CG_SoftPlus_0.75"
    else
        echo "UMPIRE_${dataset}_R${rate}_UNet_Share0_SoftPlus_${momentum}_0.75_epsilon1e-06_Smoothing_${smoothing}"
    fi
}

run_test() {
    dataset="$1"
    network="$2"
    rate="$3"
    momentum="$4"
    smoothing="$5"
    name="$(experiment_name "$dataset" "$network" "$rate" "$momentum" "$smoothing")"

    checkpoint=$(find "Train_Results/$name/model" -maxdepth 1 \
        -name "BestModel_Val_R${rate}_Epoch*.pth" 2>/dev/null \
        | sort -V | tail -n 1)

    if [ -z "$checkpoint" ]; then
        echo "Skipping $name: checkpoint not found"
        return
    fi

    epoch="${checkpoint##*Epoch}"
    epoch="${epoch%.pth}"

    python3 - "$CONFIG" "$dataset" "$network" "$rate" "$momentum" "$smoothing" "$epoch" <<'PY'
import sys, yaml

path, dataset, network, rate, momentum, smoothing, epoch = sys.argv[1:]
with open(path) as f:
    cfg = yaml.safe_load(f)

cfg["Test"]["Dataset"] = dataset
cfg["Test"]["Network"] = network
cfg["Test"]["R"] = int(rate)
cfg["Test"]["Momentum_type"] = momentum
cfg["Test"]["Smoothing_type"] = smoothing
cfg["Test"]["model_epoch"] = int(epoch)

with open(path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    echo "Testing: $name | epoch=$epoch"
    python3 -u test.py
}

# PDDL: CorPD and CorPDFS at R4, R6, and R8
for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_test "$dataset" PDDL "$rate" None Simple
    done
done

# UMPIRE: CorPD and CorPDFS at R4, R6, and R8
for dataset in CorPD CorPDFS; do
    for rate in 4 6 8; do
        run_test "$dataset" UMPIRE "$rate" Nesterov Simple
        run_test "$dataset" UMPIRE "$rate" None Simple
    done
done

# Additional CorPD R8 ablations
run_test CorPD UMPIRE 8 None LogExp
run_test CorPD UMPIRE 8 None SmoothL1
run_test CorPD UMPIRE 8 Polyak Simple
