import argparse
import csv
import inspect
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"]="1"
import shutil
from datetime import datetime
from functools import partial
import albumentations as A
import cv2
import numpy as np
import torch
from mmengine import Config
from tqdm import tqdm
from data_loader import test_dataset
import method as model_lib
from method.mssim import ssim
from utils import constructor, ops, pt_utils, py_utils
test_datasets = ['SSD','RGBD135','DUTRGBD','LFSD','SIP','NJU2K','NLPR'] 
dataset_path = 'dataset/TestDataset/'


@torch.no_grad()
def eval_once(model, data_loader, save_path=""):
    model.eval()
    bar_iter = enumerate(data_loader)
    for batch_id, batch in bar_iter:
        images = batch["image"].cuda(non_blocking=True)
        depths = batch["depth"].cuda(non_blocking=True)   
        bins = batch["bin"].cuda(non_blocking=True)   
        logits = model(data=dict(image=images, depth=depths, bin=bins))
        probs = logits.sigmoid().squeeze(1).cpu().detach().numpy()
        w = np.array(batch["w"])[0]
        h = np.array(batch["h"])[0]
        pred = cv2.resize(np.squeeze(probs), dsize=(w, h), interpolation=cv2.INTER_LINEAR)  # 0~1
        pred_name = batch["name"][0]
        ops.save_array_as_image(data_array=pred, save_name=pred_name, save_dir=save_path)
def testing(model, cfg):
    for te_data_name in test_datasets:
        save_folder='saliency_maps/SSNet/'+te_data_name
        os.makedirs(save_folder,exist_ok=True)
        root  = dataset_path + te_data_name + '/'
        te_dataset = test_dataset(root, cfg.data.test.shape["h"])
        te_loader = torch.utils.data.DataLoader(
            dataset=te_dataset,
            batch_size=1,
            num_workers=cfg.args.num_workers,
            pin_memory=True,
        )
        print(f"Testing on {te_data_name} with {len(te_loader)} samples")
        eval_once(model=model, save_path=save_folder, data_loader=te_loader)
        
def parse_config():
    ckpt = "ckpt/SSNet.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/rgbd_dataset.py")
    parser.add_argument("--model-name", type=str, default="SSNet")
    parser.add_argument("--load-from", type=str,default=ckpt)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--info", type=str)
    args = parser.parse_args()
    cfg = Config.fromfile(args.config, use_predefined_variables=False)
    cfg.merge_from_dict(vars(args))
    return cfg
def main():
    cfg = parse_config()
    pt_utils.initialize_seed_cudnn(seed=cfg.args.base_seed, deterministic=cfg.args.deterministic)
    print(f"[{datetime.now()}]")
    if hasattr(model_lib, cfg.model_name):
        ModuleClass = getattr(model_lib, cfg.model_name)
        model = ModuleClass(pretrained=cfg.pretrained)
    else:
        raise ModuleNotFoundError(f"Please add <{cfg.model_name}> into the __init__.py.")
    if cfg.load_from:
        model.load_state_dict(torch.load(cfg.load_from,weights_only=True, map_location="cpu"))
        print(f"Loaded from {cfg.load_from}")
    model.cuda()
    testing(model=model, cfg=cfg)
    print(f"{datetime.now()}: End testing...")
if __name__ == "__main__":
    main()