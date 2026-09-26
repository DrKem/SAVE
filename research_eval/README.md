# S.A.V.E. v2 — benchmark, bootstrap confidence intervals, and D_dep ablation

This folder adds a reproducible research-evaluation layer without changing the deployed S.A.V.E. application or its current production checkpoint.

## Scope

The pipeline evaluates the requested detectors on one common held-out test set:

- YOLOv11-M (`yolo11m.pt`)
- YOLOv10-M (`yolov10m.pt`)
- RT-DETR-L (`rtdetr-l.pt`)
- Faster R-CNN ResNet-50-FPN v2

It reports a common set of metrics for all models: mAP@50, mAP@50-95, macro recall at IoU 0.50 and confidence >=0.25, and batch-1 end-to-end inference latency in milliseconds (disk I/O excluded). AP is recomputed by the same repository code for every architecture rather than mixing framework-specific validators.

## 1. Create the common 70/15/15 split

The supplied Drive dataset currently contains train and validation lists but no held-out S.A.V.E. test split. On Colab, after mounting Drive:

```bash
python research_eval/prepare_dataset.py \
  --dataset-root /content/drive/MyDrive/dataset \
  --names-yaml /content/drive/MyDrive/dataset/data.yaml \
  --output-dir research_eval/generated_dataset \
  --seed 42
```

The splitter is species-stratified and groups Roboflow augmentation derivatives by source image identifier to reduce train/test leakage.

## 2. Train the four agreed models

Install research dependencies:

```bash
pip install -r research_eval/requirements.txt
```

Run the complete training/evaluation sequence on a CUDA GPU:

```bash
bash research_eval/run_benchmark.sh research_eval/generated_dataset/data.yaml
```

Or run model training individually:

```bash
python research_eval/train_ultralytics.py --model yolo11m  --data research_eval/generated_dataset/data.yaml --device 0
python research_eval/train_ultralytics.py --model yolov10m --data research_eval/generated_dataset/data.yaml --device 0
python research_eval/train_ultralytics.py --model rtdetr-l  --data research_eval/generated_dataset/data.yaml --device 0
python research_eval/train_faster_rcnn.py --train research_eval/generated_dataset/train.txt --val research_eval/generated_dataset/val.txt --device cuda
```

## 3. Common test-set benchmark and confidence intervals

`benchmark.py` exports raw test predictions, per-image latency, point estimates, and hardware metadata. AP is recomputed by one common implementation. `bootstrap_ci.py` resamples test images with replacement (B=1000, seed 42), carrying each image's ground truths, predictions, and latency together.

```bash
python research_eval/bootstrap_ci.py \
  --benchmark-dir research_eval/outputs/benchmarks/yolo11m \
  --b 1000 --seed 42
```

Repeat for each model and run `python research_eval/aggregate_results.py`.

## 4. D_dep sensitivity and ablation

```text
DeltaF = ((F_t0 - F_tn) / F_t0) * 100
DeltaC = ((C_tn - C_t0) / C_t0) * 100
D_dep  = alpha * DeltaF + (1 - alpha) * DeltaC
```

For the available count table, `F` is the summed count across matched country locations and `C` is country-level HHI spatial concentration. Alpha is varied from 0.0 to 1.0 in steps of 0.1.

- alpha=0.0: spatial only (DeltaC)
- alpha=0.5: combined spatiotemporal metric
- alpha=1.0: temporal only (DeltaF)

```bash
python research_eval/ddep_analysis.py --input data/Population_by_year.csv --output-dir research_eval/outputs
```

## Reproducibility notes

- Random seed: 42.
- Bootstrap repetitions: 1000.
- mAP@50-95 IoUs: 0.50, 0.55, ..., 0.95.
- AP uses 101-point interpolated precision.
- Recall is macro-averaged over classes present in test ground truth at IoU >=0.50 and score >=0.25.
- Latency excludes disk I/O, uses batch size 1, and excludes up to 10 warm-up images.
- Compare latency only when all models run on the same hardware/software environment.
- Existing validation results are retained only as a reference and are not presented as the requested four-model test benchmark.
