#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-research_eval/generated_dataset/data.yaml}"
DEVICE="${DEVICE:-0}"
PY="${PYTHON:-python}"
MIN_IMAGES="${MIN_IMAGES:-6500}"

$PY - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is required for the final Q1 Table 3 benchmark.")
print("CUDA:", torch.cuda.get_device_name(0))
PY

TEST=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['test'])")
TRAIN=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['train'])")
VAL=$($PY -c "import yaml; print(yaml.safe_load(open('$DATA'))['val'])")

TOTAL=$($PY - "$TRAIN" "$VAL" "$TEST" <<'PY'
import sys
n=0
for f in sys.argv[1:]:
    n += sum(1 for x in open(f,encoding="utf-8") if x.strip())
print(n)
PY
)

echo "Common split total images: $TOTAL"
if [ "$TOTAL" -lt "$MIN_IMAGES" ]; then
  echo "ERROR: benchmark split has only $TOTAL images; expected at least $MIN_IMAGES."
  echo "Refusing to produce Table 3 from a partial dataset."
  exit 2
fi

mkdir -p research_eval/outputs/benchmarks

if [ ! -f research_eval/runs/yolo11m/weights/best.pt ]; then
  $PY research_eval/train_ultralytics.py --model yolo11m --data "$DATA" --device "$DEVICE" --epochs 100 --imgsz 640 --batch 8
fi
if [ ! -f research_eval/runs/yolov10m/weights/best.pt ]; then
  $PY research_eval/train_ultralytics.py --model yolov10m --data "$DATA" --device "$DEVICE" --epochs 100 --imgsz 640 --batch 8
fi
if [ ! -f research_eval/runs/rtdetr-l/weights/best.pt ]; then
  $PY research_eval/train_ultralytics.py --model rtdetr-l --data "$DATA" --device "$DEVICE" --epochs 100 --imgsz 640 --batch 8
fi
if [ ! -f research_eval/runs/faster_rcnn/best.pt ]; then
  $PY research_eval/train_faster_rcnn.py --train "$TRAIN" --val "$VAL" --device cuda --epochs 20 --batch 4 --imgsz 640 --seed 42
fi

$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/yolo11m/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/yolo11m --device "$DEVICE" --imgsz 640
$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/yolov10m/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/yolov10m --device "$DEVICE" --imgsz 640
$PY research_eval/benchmark.py --model-type ultralytics --weights research_eval/runs/rtdetr-l/weights/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/rtdetr-l --device "$DEVICE" --imgsz 640
$PY research_eval/benchmark.py --model-type faster_rcnn --weights research_eval/runs/faster_rcnn/best.pt --test-list "$TEST" --output-dir research_eval/outputs/benchmarks/faster_rcnn --device cuda --imgsz 640

for m in yolo11m yolov10m rtdetr-l faster_rcnn; do
  $PY research_eval/bootstrap_ci.py --benchmark-dir "research_eval/outputs/benchmarks/$m" --b 1000 --seed 42
done

$PY research_eval/aggregate_results.py
echo "FINAL TABLE: research_eval/outputs/model_benchmark.csv"
