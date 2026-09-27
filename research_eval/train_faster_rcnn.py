from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from model_utils import build_faster_rcnn
from dataset_utils import label_path,parse_label_file

class DS(Dataset):
    def __init__(self,list_file):
        self.paths=[Path(x.strip()) for x in Path(list_file).read_text().splitlines() if x.strip()]
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        p=self.paths[i]
        with Image.open(p) as im:
            im=im.convert("RGB");w,h=im.size;x=pil_to_tensor(im).float()/255.0
        boxes=[];labels=[]
        for z in parse_label_file(label_path(p)):
            a,b,c,d=z["box_norm"];boxes.append([a*w,b*h,c*w,d*h]);labels.append(z["class_id"]+1)
        target={
            "boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            "labels":torch.tensor(labels,dtype=torch.int64),
            "image_id":torch.tensor([i])
        }
        return x,target

def collate(b):return tuple(zip(*b))

def seed_all(seed:int):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def loss_epoch(model,loader,device,optimizer=None):
    training=optimizer is not None
    model.train()
    # Keep BatchNorm statistics fixed during validation-loss calculation.
    if not training:
        for module in model.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
    total=0.0
    context=torch.enable_grad() if training else torch.no_grad()
    with context:
        for imgs,tgts in loader:
            imgs=[x.to(device,non_blocking=True) for x in imgs]
            tgts=[{k:v.to(device,non_blocking=True) for k,v in t.items()} for t in tgts]
            losses=model(imgs,tgts);loss=sum(losses.values())
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            total+=float(loss.detach())
    return total/max(len(loader),1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",required=True);p.add_argument("--val",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch",type=int,default=4);p.add_argument("--num-classes",type=int,default=23)
    p.add_argument("--imgsz",type=int,default=640);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--output",default="research_eval/runs/faster_rcnn")
    a=p.parse_args()

    seed_all(a.seed)
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu")
    if dev.type!="cuda":
        print("WARNING: CUDA unavailable; Faster R-CNN training will run on CPU.")

    train_ds=DS(a.train);val_ds=DS(a.val)
    g=torch.Generator();g.manual_seed(a.seed)
    train_dl=DataLoader(train_ds,batch_size=a.batch,shuffle=True,num_workers=a.workers,
                        collate_fn=collate,pin_memory=dev.type=="cuda",generator=g)
    val_dl=DataLoader(val_ds,batch_size=a.batch,shuffle=False,num_workers=a.workers,
                      collate_fn=collate,pin_memory=dev.type=="cuda")

    model=build_faster_rcnn(a.num_classes,True,a.imgsz).to(dev)
    opt=torch.optim.SGD([x for x in model.parameters() if x.requires_grad],
                        lr=.005,momentum=.9,weight_decay=.0005)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.epochs//3),gamma=.1)

    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    history=[];best=float("inf")
    for epoch in range(a.epochs):
        train_loss=loss_epoch(model,train_dl,dev,opt)
        val_loss=loss_epoch(model,val_dl,dev,None)
        lr=float(opt.param_groups[0]["lr"])
        row={"epoch":epoch+1,"train_loss":train_loss,"val_loss":val_loss,"lr":lr}
        history.append(row);print(row)
        payload={
            "model":model.state_dict(),"num_classes":a.num_classes,"imgsz":a.imgsz,
            "epoch":epoch+1,"seed":a.seed,"val_loss":val_loss
        }
        torch.save(payload,out/"last.pt")
        if val_loss<best:
            best=val_loss;torch.save(payload,out/"best.pt")
        sch.step()
        (out/"history.json").write_text(json.dumps(history,indent=2))
    print(f"Best validation loss: {best:.6f}")

if __name__=="__main__":main()
