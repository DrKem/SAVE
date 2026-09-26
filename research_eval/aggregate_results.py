from __future__ import annotations
import argparse,csv,json
from pathlib import Path
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default="research_eval/outputs/benchmarks");p.add_argument("--output",default="research_eval/outputs/model_benchmark.csv");a=p.parse_args()
    rows=[]
    for ci_file in sorted(Path(a.root).glob("*/bootstrap_ci.json")):
        d=ci_file.parent;ci=json.loads(ci_file.read_text());summary=json.loads((d/"summary.json").read_text())
        row={"model":d.name,"n_test_images":summary.get("n_images"),"device":summary.get("device"),"gpu":summary.get("environment",{}).get("gpu")}
        for k,v in ci["metrics"].items():row[k]=v["estimate"];row[k+"_ci95_low"]=v["ci95_low"];row[k+"_ci95_high"]=v["ci95_high"]
        rows.append(row)
    if not rows:raise SystemExit("No benchmark CI results found")
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(out)
if __name__=="__main__":main()
