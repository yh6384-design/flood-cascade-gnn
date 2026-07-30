#!/usr/bin/env bash
# Reproduce everything from a clean checkout. ~10 minutes on a laptop CPU.
set -euo pipefail
export PYTHONPATH=src

echo "[1/3] simulating flood event set"
python src/simulate.py --events 3000 --seed 42 --out data/scenarios.npz

echo "[2/3] training GNN and MLP baseline"
python src/train.py --data data/scenarios.npz --outdir artifacts \
                    --epochs 25 --batch-size 128

echo "[3/3] rendering figures"
python src/figures.py --data data/scenarios.npz \
                     --preds artifacts/test_predictions.npz \
                     --outdir figures

echo "done. see artifacts/metrics.json and figures/"
