#!/bin/bash

set -e

echo "========================================"
echo " FakeShield / WebWarriors"
echo "========================================"
echo

echo "===== STEP 1: BUILD / UPDATE SUBSET ====="
python3 webwarriors/baseline/script/build_subset.py

echo
echo "===== STEP 2: DTE-FDM ====="

docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/ml_project/FakeShield/DTE-FDM \
  -v /mnt/DATADRIVE0/Rajnish_B24DS024/ml_project:/workspace/ml_project \
  -w /workspace/ml_project/FakeShield \
  zhipeixu/dte-fdm:v1.0 \
  python3 webwarriors/baseline/script/run_dte_fdm.py

echo
echo "===== STEP 3: MFLM ====="

docker run --rm --gpus all \
  -e PYTHONPATH=/workspace/ml_project/FakeShield/MFLM \
  -v /mnt/DATADRIVE0/Rajnish_B24DS024/ml_project:/workspace/ml_project \
  -w /workspace/ml_project/FakeShield \
  fakeshield-mflm-runtime:1.1 \
  python3 webwarriors/baseline/script/run_mflm.py

echo
echo "===== STEP 4: CALCULATE METRICS ====="

python3 webwarriors/baseline/script/calculate_metrics.py \
  --output "./webwarriors/basline/results/metric/metrics_$(date +%Y%m%d_%H%M%S).csv"

echo
echo "========================================"
echo " PHASE 2 COMPLETE"
echo "========================================"
