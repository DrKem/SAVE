from __future__ import annotations
import argparse,csv,json,re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

AGGREGATE_PATTERNS=[re.compile(x,re.I) for x in [r"^world$",r"africa\(",r"^central africa$",r"^eastern africa$",r"^southern africa$",r"^western africa$"]]

def aggregate_location(name):
    return any(x.search(name.strip()) for x in AGGREGATE_PATTERNS)

def read_rows(path):
    rows=[]
    with open(path,encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            try:rows.append({"species":r["Animal"].strip(),"location":r["Country"].strip(),"year":int(r["Year"]),"count":float(r["Population"])})
            except (ValueError,KeyError):pass
    return rows

def hhi(values):
    s=sum(values)
    if s<=0:return None
    return sum((v/s)**2 for v in values)

def endpoints(rows,min_locations=2):
    out=[]
    for sp in sorted({r["species"] for r in rows}):
        by=defaultdict(list)
        for r in rows:
            if r["species"]==sp and not aggregate_location(r["location"]):by[r["year"]].append(r)
        years=sorted(y for y,v in by.items() if len(v)>=min_locations);best=None
        for i,y0 in enumerate(years):
            for y1 in years[i+1:]:
                m0={x["location"]:x["count"] for x in by[y0]};m1={x["location"]:x["count"] for x in by[y1]};common=sorted(set(m0)&set(m1))
                if len(common)<min_locations:continue
                candidate=(y1-y0,len(common),y0,y1,common,m0,m1)
                if best is None or candidate[:2]>best[:2]:best=candidate
        if best is None:continue
        _,_,y0,y1,common,m0,m1=best;p0=[m0[k] for k in common];p1=[m1[k] for k in common];f0=sum(p0);f1=sum(p1);c0=hhi(p0);c1=hhi(p1)
        if not f0 or not f1 or c0 is None or c1 is None:continue
        df=(f0-f1)/f0*100;dc=(c1-c0)/c0*100
        out.append({"species":sp,"baseline_year":y0,"final_year":y1,"common_locations":len(common),"locations":common,"F_t0":f0,"F_tn":f1,"C_t0":c0,"C_tn":c1,"delta_F_pct":df,"delta_C_pct":dc})
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output-dir",default="research_eval/outputs");p.add_argument("--min-locations",type=int,default=2);a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);plots=out/"ddep_plots";plots.mkdir(exist_ok=True)
    base=endpoints(read_rows(a.input),a.min_locations);sensitivity=[];ablation=[]
    for x in base:
        core={k:v for k,v in x.items() if k!="locations"}
        for i in range(11):
            alpha=i/10;sensitivity.append({**core,"alpha":alpha,"D_dep":alpha*x["delta_F_pct"]+(1-alpha)*x["delta_C_pct"]})
        for component,alpha in [("spatial_only",0.0),("combined",0.5),("temporal_only",1.0)]:
            ablation.append({**core,"component":component,"alpha":alpha,"D_dep":alpha*x["delta_F_pct"]+(1-alpha)*x["delta_C_pct"]})
        xs=[i/10 for i in range(11)];ys=[z*x["delta_F_pct"]+(1-z)*x["delta_C_pct"] for z in xs]
        fig,ax=plt.subplots(figsize=(7,4.5));ax.plot(xs,ys,marker="o");ax.axhline(0,linewidth=.8);ax.set_xlabel("alpha (temporal weight)");ax.set_ylabel("D_dep (%)");ax.set_title(x["species"]);fig.tight_layout()
        slug=re.sub(r"[^a-z0-9]+","_",x["species"].lower()).strip("_");fig.savefig(plots/f"{slug}.png",dpi=180);plt.close(fig)
    def write(name,rows):
        if rows:
            with (out/name).open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write("ddep_sensitivity.csv",sensitivity);write("ddep_ablation.csv",ablation)
    (out/"ddep_endpoints.json").write_text(json.dumps(base,indent=2))
    method={"formula":"D_dep = alpha*DeltaF + (1-alpha)*DeltaC","DeltaF":"((F_t0-F_tn)/F_t0)*100","DeltaC":"((C_tn-C_t0)/C_t0)*100","C":"HHI across matched country locations = sum((count_i/F_t)^2)","endpoint_rule":"longest interval with >= min_locations country locations observed at both endpoints; aggregate/world rows excluded","important":"Country concentration is an available-data spatial proxy, not a camera-coordinate clustering statistic."}
    (out/"ddep_method.json").write_text(json.dumps(method,indent=2))
    print(f"{len(base)} species written to {out}")
if __name__=="__main__":main()
