from __future__ import annotations

import argparse, json, platform, time
from pathlib import Path
import torch
from PIL import Image
from dataset_utils import save_json
from metrics import evaluate
from model_utils import build_faster_rcnn, gt_for_image, image_tensor

def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize()

def load_ultralytics(weights):
    from ultralytics import YOLO, RTDETR
    return RTDETR(weights) if "rtdetr" in Path(weights).name.lower() else YOLO(weights)

def predict_ultralytics(model,image,device,imgsz):
    result=model.predict(source=image,imgsz=imgsz,conf=0.001,verbose=False,device=device)[0]
    if result.boxes is None:return []
    boxes=result.boxes.xyxy.detach().cpu().numpy();scores=result.boxes.conf.detach().cpu().numpy();classes=result.boxes.cls.detach().cpu().numpy().astype(int)
    return [{"class_id":int(c),"score":float(s),"box":[float(x) for x in b]} for b,s,c in zip(boxes,scores,classes)]

def predict_faster(model,image_path,device):
    x=image_tensor(image_path).to(device);result=model([x])[0]
    boxes=result["boxes"].detach().cpu().numpy();scores=result["scores"].detach().cpu().numpy();labels=result["labels"].detach().cpu().numpy().astype(int)
    return [{"class_id":int(c)-1,"score":float(s),"box":[float(x) for x in b]} for b,s,c in zip(boxes,scores,labels) if c>0]

def environment():
    return {"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

def main():
    p=argparse.ArgumentParser();p.add_argument("--model-type",required=True,choices=["ultralytics","faster_rcnn"]);p.add_argument("--weights",required=True);p.add_argument("--test-list",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--imgsz",type=int,default=640);p.add_argument("--num-classes",type=int,default=23);p.add_argument("--warmup",type=int,default=10);a=p.parse_args()
    paths=[Path(x.strip()) for x in Path(a.test_list).read_text().splitlines() if x.strip()]
    if not paths:raise SystemExit("Empty test list")
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.model_type=="ultralytics":
        model=load_ultralytics(a.weights);device=a.device
        for path in paths[:min(a.warmup,len(paths))]:
            with Image.open(path) as im:predict_ultralytics(model,im.convert("RGB"),device,a.imgsz)
        sync(device)
    else:
        device=a.device if torch.cuda.is_available() or not a.device.startswith("cuda") else "cpu"
        ckpt=torch.load(a.weights,map_location=device);n=int(ckpt.get("num_classes",a.num_classes)) if isinstance(ckpt,dict) else a.num_classes
        model=build_faster_rcnn(n,pretrained=False);state=ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt;model.load_state_dict(state);model.to(device);model.eval()
        with torch.inference_mode():
            for path in paths[:min(a.warmup,len(paths))]:predict_faster(model,path,device)
        sync(device)
    records=[]
    for i,path in enumerate(paths,1):
        gt,w,h=gt_for_image(path)
        if a.model_type=="ultralytics":
            with Image.open(path) as im:
                rgb=im.convert("RGB");sync(device);t0=time.perf_counter();preds=predict_ultralytics(model,rgb,device,a.imgsz);sync(device);elapsed=(time.perf_counter()-t0)*1000
        else:
            with torch.inference_mode():
                sync(device);t0=time.perf_counter();preds=predict_faster(model,path,device);sync(device);elapsed=(time.perf_counter()-t0)*1000
        records.append({"image":str(path),"width":w,"height":h,"ground_truth":gt,"predictions":preds,"latency_ms":elapsed})
        if i%100==0:print(f"{i}/{len(paths)}")
    summary=evaluate(records);meta={"model_type":a.model_type,"weights":str(Path(a.weights).resolve()),"imgsz":a.imgsz,"device":device,"warmup_images":min(a.warmup,len(paths)),"environment":environment()}
    save_json(out/"predictions.json",records);save_json(out/"summary.json",{**summary,**meta});print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
