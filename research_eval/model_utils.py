from __future__ import annotations
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from dataset_utils import label_path, parse_label_file

def gt_for_image(path:Path):
    with Image.open(path) as im:w,h=im.size
    out=[]
    for x in parse_label_file(label_path(path)):
        a,b,c,d=x["box_norm"];out.append({"class_id":x["class_id"],"box":[a*w,b*h,c*w,d*h]})
    return out,w,h

def build_faster_rcnn(num_classes:int,pretrained:bool=True):
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model=fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes+1)
    return model

def image_tensor(path:Path):
    with Image.open(path) as im:return pil_to_tensor(im.convert("RGB")).float()/255.0
