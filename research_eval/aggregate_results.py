from __future__ import annotations
import argparse,csv,json
from pathlib import Path

MODEL_META={
    "yolo11m":{
        "architecture":"YOLOv11-M",
        "paradigm":"One-stage real-time detector",
        "role":"Primary one-stage edge/deployment candidate",
    },
    "yolov10m":{
        "architecture":"YOLOv10-M",
        "paradigm":"One-stage real-time detector",
        "role":"Efficient one-stage real-time comparator",
    },
    "rtdetr-l":{
        "architecture":"RT-DETR-L",
        "paradigm":"Real-time transformer detector",
        "role":"Transformer-based real-time comparator",
    },
    "faster_rcnn":{
        "architecture":"Faster R-CNN ResNet-50-FPN v2",
        "paradigm":"Two-stage detector",
        "role":"Accuracy-oriented two-stage reference",
    },
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="research_eval/outputs/benchmarks")
    p.add_argument("--output",default="research_eval/outputs/model_benchmark.csv")
    a=p.parse_args()

    rows=[]
    for ci_file in sorted(Path(a.root).glob("*/bootstrap_ci.json")):
        d=ci_file.parent
        ci=json.loads(ci_file.read_text())
        summary=json.loads((d/"summary.json").read_text())
        meta=MODEL_META.get(d.name,{
            "architecture":d.name,"paradigm":"","role":""
        })
        row={
            "model":d.name,
            "detection_architecture":meta["architecture"],
            "model_category_paradigm":meta["paradigm"],
            "footprint_mb":summary.get("footprint_mb"),
            "precision":ci["metrics"]["precision"]["estimate"],
            "precision_ci95_low":ci["metrics"]["precision"]["ci95_low"],
            "precision_ci95_high":ci["metrics"]["precision"]["ci95_high"],
            "recall":ci["metrics"]["recall"]["estimate"],
            "recall_ci95_low":ci["metrics"]["recall"]["ci95_low"],
            "recall_ci95_high":ci["metrics"]["recall"]["ci95_high"],
            "mAP50":ci["metrics"]["mAP50"]["estimate"],
            "mAP50_ci95_low":ci["metrics"]["mAP50"]["ci95_low"],
            "mAP50_ci95_high":ci["metrics"]["mAP50"]["ci95_high"],
            "mAP50_95":ci["metrics"]["mAP50_95"]["estimate"],
            "mAP50_95_ci95_low":ci["metrics"]["mAP50_95"]["ci95_low"],
            "mAP50_95_ci95_high":ci["metrics"]["mAP50_95"]["ci95_high"],
            "latency_ms":ci["metrics"]["latency_ms"]["estimate"],
            "latency_ms_ci95_low":ci["metrics"]["latency_ms"]["ci95_low"],
            "latency_ms_ci95_high":ci["metrics"]["latency_ms"]["ci95_high"],
            "n_test_images":summary.get("n_images"),
            "device":summary.get("device"),
            "gpu":summary.get("environment",{}).get("gpu"),
            "operational_role":meta["role"],
        }
        rows.append(row)

    if not rows:raise SystemExit("No benchmark CI results found")
    if len(rows)!=4:
        raise SystemExit(f"Expected 4 completed model benchmarks, found {len(rows)}")

    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print(out)

if __name__=="__main__":main()
