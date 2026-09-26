# Benchmark execution status

The shared Drive contains a 23-class consolidated dataset with 5,461 entries in train.txt and 1,166 in valid.txt. Its original data.yaml does not define a held-out test set. The existing model_metrics/results.csv belongs to one prior training run and therefore cannot serve as the requested four-model test comparison.

This branch creates one leakage-resistant 70/15/15 split, trains all four agreed architectures, evaluates raw predictions with the same metric implementation, measures batch-1 latency on the same machine, and performs image-level percentile bootstrap resampling with B=1000.

The numerical four-model table is intentionally not pre-filled. Actual values must be produced by executing the GPU benchmark because the repository/Drive do not currently contain checkpoints and test predictions for all four requested models. Published COCO values and the existing single-model validation CSV are not substituted.

After execution, the populated table is generated at research_eval/outputs/model_benchmark.csv.
