from __future__ import annotations

import json, random, re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
RF_RE=re.compile(r"_jpg\.rf\.[0-9a-f]+$",re.I)

def source_group(path:Path, root:Path)->str:
    rel=path.relative_to(root)
    species=rel.parts[0] if rel.parts else "unknown"
    stem=RF_RE.sub("",path.stem)
    return f"{species}::{stem.lower()}"

def scan_dataset(root:Path)->list[Path]:
    return sorted(p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and ("train" in p.parts or "valid" in p.parts) and "images" in p.parts)

def label_path(image:Path)->Path:
    parts=list(image.parts)
    try:i=len(parts)-1-parts[::-1].index("images");parts[i]="labels"
    except ValueError:return image.with_suffix(".txt")
    return Path(*parts).with_suffix(".txt")

def parse_label_file(path:Path)->list[dict]:
    out=[]
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        t=line.strip().split()
        if len(t)<5: continue
        cls=int(float(t[0])); vals=[float(x) for x in t[1:]]
        if len(vals)==4:
            cx,cy,w,h=vals; box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
        elif len(vals)>=6 and len(vals)%2==0:
            xs=vals[0::2];ys=vals[1::2];box=[min(xs),min(ys),max(xs),max(ys)]
        else: continue
        out.append({"class_id":cls,"box_norm":[max(0,box[0]),max(0,box[1]),min(1,box[2]),min(1,box[3])]})
    return out

def image_species(path:Path,root:Path)->str:
    try:return path.relative_to(root).parts[0]
    except Exception:return "unknown"

def grouped_stratified_split(images:list[Path],root:Path,train_frac:float,val_frac:float,seed:int)->dict[str,list[Path]]:
    if train_frac<=0 or val_frac<0 or train_frac+val_frac>=1: raise ValueError("train/val fractions must leave a positive test fraction")
    groups=defaultdict(list)
    for p in images: groups[(image_species(p,root),source_group(p,root))].append(p)
    by_species=defaultdict(list)
    for (sp,g),paths in groups.items(): by_species[sp].append((g,paths))
    rng=random.Random(seed); out={"train":[],"val":[],"test":[]}
    for sp, gs in sorted(by_species.items()):
        rng.shuffle(gs); n=sum(len(x[1]) for x in gs); targets={"train":train_frac*n,"val":val_frac*n,"test":(1-train_frac-val_frac)*n}; counts={k:0 for k in out}
        for g,paths in sorted(gs,key=lambda x:len(x[1]),reverse=True):
            deficits={k:targets[k]-counts[k] for k in out}; split=max(deficits,key=deficits.get)
            out[split].extend(paths); counts[split]+=len(paths)
    for k in out: out[k]=sorted(out[k])
    return out

def write_split_file(path:Path,paths:Iterable[Path])->None:
    path.write_text("\n".join(str(p) for p in paths)+"\n",encoding="utf-8")

def save_json(path:Path,obj)->None:
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2),encoding="utf-8")

def split_manifest(splits:dict[str,list[Path]],root:Path)->dict:
    d={"total":sum(map(len,splits.values())),"splits":{}}
    for k,paths in splits.items():
        d["splits"][k]={"images":len(paths),"source_groups":len({source_group(p,root) for p in paths}),"species":dict(sorted(__import__("collections").Counter(image_species(p,root) for p in paths).items()))}
    return d
