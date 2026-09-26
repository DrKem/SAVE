from __future__ import annotations
import argparse
from ultralytics import YOLO, RTDETR

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",required=True,choices=["yolo11m","yolov10m","rtdetr-l"]);p.add_argument("--data",required=True);p.add_argument("--device",default="0");p.add_argument("--epochs",type=int,default=100);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--batch",type=int,default=8);a=p.parse_args()
    m=RTDETR("rtdetr-l.pt") if a.model=="rtdetr-l" else YOLO(a.model+".pt")
    m.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,seed=42,deterministic=True,project="research_eval/runs",name=a.model,exist_ok=True)
if __name__=="__main__":main()
