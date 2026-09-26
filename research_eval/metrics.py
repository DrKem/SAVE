from __future__ import annotations
import math
import numpy as np

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-12)

def _ap101(rec,prec):
    if len(rec)==0:return 0.0
    grid=np.linspace(0,1,101); vals=[]
    for r in grid:
        m=prec[rec>=r];vals.append(float(m.max()) if len(m) else 0.0)
    return float(np.mean(vals))

def class_ap(records,cls,iou_thr):
    n_gt=0; preds=[]; gt_by={}
    for idx,r in enumerate(records):
        g=[x["box"] for x in r["ground_truth"] if x["class_id"]==cls];gt_by[idx]=g;n_gt+=len(g)
        for p in r["predictions"]:
            if p["class_id"]==cls: preds.append((float(p["score"]),idx,p["box"]))
    if n_gt==0:return math.nan
    preds.sort(reverse=True,key=lambda x:x[0]); matched={i:set() for i in gt_by};tp=[];fp=[]
    for score,idx,box in preds:
        best=-1;best_iou=0.0
        for j,g in enumerate(gt_by[idx]):
            if j in matched[idx]:continue
            v=iou(box,g)
            if v>best_iou:best_iou=v;best=j
        if best>=0 and best_iou>=iou_thr:matched[idx].add(best);tp.append(1);fp.append(0)
        else:tp.append(0);fp.append(1)
    if not preds:return 0.0
    tp=np.cumsum(tp);fp=np.cumsum(fp);rec=tp/n_gt;prec=tp/np.maximum(tp+fp,1)
    return _ap101(rec,prec)

def macro_recall(records,score_thr=.25,iou_thr=.5):
    classes=sorted({g["class_id"] for r in records for g in r["ground_truth"]}); vals=[]
    for cls in classes:
        gt_total=0;tp=0
        for r in records:
            g=[x["box"] for x in r["ground_truth"] if x["class_id"]==cls];gt_total+=len(g);used=set()
            ps=sorted([p for p in r["predictions"] if p["class_id"]==cls and p["score"]>=score_thr],key=lambda x:x["score"],reverse=True)
            for p in ps:
                best=-1;bv=0
                for j,b in enumerate(g):
                    if j in used:continue
                    v=iou(p["box"],b)
                    if v>bv:bv=v;best=j
                if best>=0 and bv>=iou_thr:used.add(best);tp+=1
        if gt_total:vals.append(tp/gt_total)
    return float(np.mean(vals)) if vals else math.nan

def evaluate(records):
    classes=sorted({g["class_id"] for r in records for g in r["ground_truth"]})
    ious=[round(x,2) for x in np.arange(.5,.951,.05)]
    by_iou=[]
    for t in ious:
        aps=[class_ap(records,c,t) for c in classes];aps=[x for x in aps if not math.isnan(x)]
        by_iou.append(float(np.mean(aps)) if aps else math.nan)
    lat=[float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    return {"mAP50":by_iou[0],"mAP50_95":float(np.mean(by_iou)),"recall":macro_recall(records),"latency_ms":float(np.mean(lat)) if lat else math.nan,"n_images":len(records),"n_classes_gt":len(classes)}
