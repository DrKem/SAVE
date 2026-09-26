from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from model_utils import build_faster_rcnn
from dataset_utils import label_path,parse_label_file

class DS(Dataset):
    def __init__(self,list_file):self.paths=[Path(x.strip()) for x in Path(list_file).read_text().splitlines() if x.strip()]
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        p=self.paths[i]
        with Image.open(p) as im:
            im=im.convert("RGB");w,h=im.size;x=pil_to_tensor(im).float()/255
        boxes=[];labels=[]
        for z in parse_label_file(label_path(p)):
            a,b,c,d=z["box_norm"];boxes.append([a*w,b*h,c*w,d*h]);labels.append(z["class_id"]+1)
        target={"boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),"labels":torch.tensor(labels,dtype=torch.int64),"image_id":torch.tensor([i])}
        return x,target
def collate(b):return tuple(zip(*b))

def main():
    p=argparse.ArgumentParser();p.add_argument("--train",required=True);p.add_argument("--val",required=True);p.add_argument("--device",default="cuda");p.add_argument("--epochs",type=int,default=20);p.add_argument("--batch",type=int,default=4);p.add_argument("--num-classes",type=int,default=23);p.add_argument("--output",default="research_eval/runs/faster_rcnn");a=p.parse_args()
    dev=torch.device(a.device if torch.cuda.is_available() else "cpu");model=build_faster_rcnn(a.num_classes,True).to(dev)
    dl=DataLoader(DS(a.train),batch_size=a.batch,shuffle=True,num_workers=2,collate_fn=collate)
    opt=torch.optim.SGD([x for x in model.parameters() if x.requires_grad],lr=.005,momentum=.9,weight_decay=.0005)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=max(1,a.epochs//3),gamma=.1)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(a.epochs):
        model.train();total=0
        for imgs,tgts in dl:
            imgs=[x.to(dev) for x in imgs];tgts=[{k:v.to(dev) for k,v in t.items()} for t in tgts]
            losses=model(imgs,tgts);loss=sum(losses.values());opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())
        sch.step();mean=total/max(len(dl),1);history.append({"epoch":epoch+1,"train_loss":mean});print(history[-1])
        torch.save({"model":model.state_dict(),"num_classes":a.num_classes,"epoch":epoch+1},out/"last.pt")
    (out/"history.json").write_text(json.dumps(history,indent=2))
if __name__=="__main__":main()
