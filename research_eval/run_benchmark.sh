#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-research_eval/generated_dataset/data.yaml}"
DEVICE="${DEVICE:-0}"
PY="${PYTHON:-python}"
TEST=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['test'])")
TRAIN=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['train'])")
VAL=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['val'])")
mkdir -p research_eval/outputs/benchmarks
$PY research_eval/train_ultralytics.py --model yolo11m --data "$DATA" --device "$DEVICE"
$PY research_eval/train_ultralytics.py --model yolov10m --data "$DATA" --device "$DEVICE"
$PY research_eval/train_ultralytics.py --model rtdetr-l --data "$DATA" --device "$DEVICE"
$PY research_eval/train_faster_rcnn.py --train "$TRAIN" --val "$VAL" --device cuda
$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/yolo11m/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/yolo11m --device "$DEVICE"
$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/yolov10m/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/yolov10m --device "$DEVICE"
$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/rtdetr-l/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/rtdetr-l --device "$DEVICE"
$PY research_eval/benchmark.py --model-type faster_rcnn --weights research_eval/runs/faster_rcnn/last.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/faster_rcnn --device cuda
for m in yolo11m yolov10m rtdetr-l faster_rcnn; do
  $PY research_eval/bootstrap_ci.py --benchmark-dir "research_eval/outputs/benchmarks/$m" --b 1000 --seed 42
done
$PY research_eval/aggregate_results.py
$PY research_eval/ddep_analysis.py --input data/Population_by_year.csv --output-dir research_eval/outputs
