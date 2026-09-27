# S.A.V.E. v2 — final Q1 detector benchmark

This folder contains the reproducible evaluation pipeline for Reviewer Comment 5. It does not alter the deployed S.A.V.E. application or archived production checkpoint.

## Models

The final benchmark evaluates the four requested architectures on one common held-out test set:

- YOLOv11-M (`yolo11m.pt`)
- YOLOv10-M (`yolov10m.pt`)
- RT-DETR-L (`rtdetr-l.pt`)
- Faster R-CNN ResNet-50-FPN v2

## Common metrics

Every architecture is scored by the same repository evaluator:

- macro precision at IoU >= 0.50 and confidence >= 0.25
- macro recall at IoU >= 0.50 and confidence >= 0.25
- mAP@50
- mAP@50-95, using IoU thresholds 0.50:0.05:0.95
- checkpoint footprint in decimal MB
- batch-1 end-to-end inference latency in milliseconds, excluding disk I/O and excluding warm-up images

AP uses 101-point interpolation. Latency comparisons are valid only when all four models are evaluated on the same GPU/software environment.

## 1. Create the leakage-resistant 70/15/15 split

The archived Drive source contains train and validation material but no independent held-out test partition. Create a new grouped, species-stratified split with seed 42:

```bash
python research_eval/prepare_dataset.py \
  --dataset-root /content/drive/MyDrive/SAVE_dataset \
  --names-yaml /content/drive/MyDrive/SAVE_dataset/data.yaml \
  --output-dir research_eval/generated_dataset \
  --seed 42 \
  --min-images 6500
```

The `--min-images 6500` guard prevents the final Table 3 from being generated from an incomplete Drive copy. Roboflow augmentation derivatives are grouped by source image identifier to reduce train/test leakage.

## 2. Train and evaluate all four models

A CUDA GPU is required for the final Table 3 run.

```bash
pip install -r research_eval/requirements.txt
bash research_eval/run_benchmark.sh research_eval/generated_dataset/data.yaml
```

Training protocol:

- YOLOv11-M: 100 epochs, 640 px, batch 8
- YOLOv10-M: 100 epochs, 640 px, batch 8
- RT-DETR-L: 100 epochs, 640 px, batch 8
- Faster R-CNN ResNet-50-FPN v2: 20 epochs, 640 px internal transform, batch 4
- deterministic seed: 42
- Faster R-CNN selects the checkpoint with lowest validation loss

The run script refuses to produce the final table if the common split contains fewer than 6,500 images.

## 3. Common test-set benchmark and uncertainty

`benchmark.py` saves:

- `predictions.json`: per-image ground truth, predictions and latency
- `summary.json`: point metrics, model footprint and hardware/software metadata

For each model, `bootstrap_ci.py` performs B=1000 non-parametric bootstrap resamples with seed 42 and reports percentile 95% confidence intervals for precision, recall, mAP@50, mAP@50-95 and latency.

```bash
python research_eval/bootstrap_ci.py \
  --benchmark-dir research_eval/outputs/benchmarks/yolo11m \
  --b 1000 --seed 42
```

The complete run automatically repeats this for all four models.

## 4. Final Table 3 output

After all four models have completed:

```text
research_eval/outputs/model_benchmark.csv
```

The CSV includes architecture, paradigm, footprint, precision, recall, mAP@50, mAP@50-95, latency, all associated 95% confidence intervals, test sample count, GPU metadata and operational deployment role.

The aggregator fails if fewer than four completed benchmark outputs are present. This prevents placeholder or partial rows from being mistaken for experimental results.

## 5. D_dep sensitivity and ablation

```text
DeltaF = ((F_t0 - F_tn) / F_t0) * 100
DeltaC = ((C_tn - C_t0) / C_t0) * 100
D_dep  = alpha * DeltaF + (1 - alpha) * DeltaC
```

For the available count table, `F` is the summed count across matched country locations and `C` is country-level HHI spatial concentration. Alpha is varied from 0.0 to 1.0 in steps of 0.1.

```bash
python research_eval/ddep_analysis.py --input data/Population_by_year.csv --output-dir research_eval/outputs
```

## Reproducibility notes

- split seed: 42
- training seed: 42
- bootstrap seed: 42
- bootstrap repetitions: 1000
- common test set for all four detectors
- common metric implementation for all four detectors
- same GPU required for latency comparison
- archived YOLO11n validation numbers are retained only as a historical S.A.V.E. baseline and are not substituted for the requested four-model held-out test benchmark
