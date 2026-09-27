from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from dataset_utils import grouped_stratified_split,save_json,scan_dataset,split_manifest,write_split_file

DEFAULT_NAMES=[
    "Alligator","Beetle","Butterfly","Corals","Fish","Frog","Gorilla","Kiwi",
    "Ladybug","Lion","Pandas","Pangolin","Projeto Macaoba","Rabbit","Salmon",
    "Sea Cucumber","Sea Turtle","Seahorse","Snail","Sphyma","Spider","Toad","Tuna Fish",
]

def main()->None:
    ap=argparse.ArgumentParser(description="Create a leakage-resistant 70/15/15 S.A.V.E. split.")
    ap.add_argument("--dataset-root",required=True)
    ap.add_argument("--output-dir",default="research_eval/generated_dataset")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--train",type=float,default=0.70)
    ap.add_argument("--val",type=float,default=0.15)
    ap.add_argument("--names-yaml")
    ap.add_argument("--min-images",type=int,default=0,
                    help="Fail if fewer source images are found; use 6500 for the final archived S.A.V.E. benchmark.")
    args=ap.parse_args()

    root=Path(args.dataset_root).resolve()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)

    images=scan_dataset(root)
    if not images:raise SystemExit(f"No images found under {root}")
    if args.min_images and len(images)<args.min_images:
        raise SystemExit(
            f"Only {len(images)} images found under {root}; expected at least {args.min_images}. "
            "Refusing to create a final benchmark split from a partial dataset."
        )

    splits=grouped_stratified_split(images,root,args.train,args.val,args.seed)
    for split,paths in splits.items():write_split_file(out/f"{split}.txt",paths)

    names=DEFAULT_NAMES
    if args.names_yaml:
        src=yaml.safe_load(Path(args.names_yaml).read_text(encoding="utf-8-sig"))
        candidate=src.get("names")
        if isinstance(candidate,dict):names=[candidate[i] for i in sorted(candidate)]
        elif isinstance(candidate,list):names=candidate

    data_yaml={
        "train":str((out/"train.txt").resolve()),
        "val":str((out/"val.txt").resolve()),
        "test":str((out/"test.txt").resolve()),
        "nc":len(names),
        "names":names,
    }
    (out/"data.yaml").write_text(yaml.safe_dump(data_yaml,sort_keys=False),encoding="utf-8")

    manifest=split_manifest(splits,root)
    manifest.update({
        "seed":args.seed,"train_fraction":args.train,"val_fraction":args.val,
        "test_fraction":1-args.train-args.val,
    })
    save_json(out/"split_manifest.json",manifest)

    print(f"Created {manifest['total']} image split at {out}")
    for split,meta in manifest["splits"].items():
        print(f"  {split}: {meta['images']} images, {meta['source_groups']} source groups")

if __name__=="__main__":main()
