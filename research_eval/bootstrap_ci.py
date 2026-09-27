from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from metrics import evaluate

METRICS=["precision","recall","mAP50","mAP50_95","latency_ms"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--benchmark-dir",required=True)
    p.add_argument("--b",type=int,default=1000)
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()

    d=Path(a.benchmark_dir)
    records=json.loads((d/"predictions.json").read_text())
    if not records:raise SystemExit("No predictions")
    point=evaluate(records);rng=np.random.default_rng(a.seed)
    samples={k:[] for k in METRICS};n=len(records)
    for i in range(a.b):
        idx=rng.integers(0,n,size=n)
        m=evaluate([records[j] for j in idx])
        for k in METRICS:samples[k].append(m[k])
        if (i+1)%100==0:print(f"bootstrap {i+1}/{a.b}")

    ci={
        k:{
            "estimate":float(point[k]),
            "ci95_low":float(np.nanpercentile(samples[k],2.5)),
            "ci95_high":float(np.nanpercentile(samples[k],97.5),
            )
        } for k in METRICS
    }
    payload={
        "bootstrap_repetitions":a.b,"seed":a.seed,
        "sampling_unit":"held-out test image",
        "method":"non-parametric percentile bootstrap",
        "precision_recall_threshold":"IoU >= 0.50 and confidence >= 0.25",
        "metrics":ci,
    }
    (d/"bootstrap_ci.json").write_text(json.dumps(payload,indent=2))
    with (d/"bootstrap_samples.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["iteration"]+METRICS);w.writeheader()
        for i in range(a.b):
            w.writerow({"iteration":i+1,**{k:samples[k][i] for k in METRICS}})
    print(json.dumps(payload,indent=2))

if __name__=="__main__":main()
