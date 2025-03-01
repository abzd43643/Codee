import torch
import os
from utils.data import get_data_from_txt, get_datasets_info_with_keys, read_binary_array, read_color_array
import cv2
import albumentations as A
from skimage.filters import threshold_multiotsu
import numpy as np

def otsu_map(depth,nb=2):
    regions = np.digitize(depth, bins=threshold_multiotsu(depth,nb))
    bin = regions == 0
    return bin 

def contrast(img,size_n=5):
    img_max = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((size_n, size_n)))

    img_min = cv2.morphologyEx(img, cv2.MORPH_ERODE, np.ones((size_n, size_n)))

    img_max = img_max.astype(float)
    img_min = img_min.astype(float)

    img_contrast = (img_max - img_min) / (img_max + img_min+1e-8)
    return img_contrast

def normm(img):
    dmi= np.min(img)
    dma=np.max(img)
    img = (img-dmi)/(dma-dmi+1e-8)
    return img
    
def central_mapp(img):
    hh,ww=img.shape
    coords_h = np.arange(-hh//2,hh//2+1)
    coords_w = np.arange(-ww//2,ww//2+1)
    y,x = np.meshgrid(coords_h, coords_w, indexing='ij')
    diss=np.sqrt(y**2+x**2)
    diss=np.max(diss)-diss
    diss = np.delete(diss, hh//2, 0)
    diss = np.delete(diss, ww//2, 1)
    dmi= np.min(diss)
    dma=np.max(diss)
    diss = (diss-dmi)/(dma-dmi+1e-8)
    return diss  

def saliency_priors(rgb,depth,otsu,n_size=5):
    rgb = np.array(rgb)
    depth = np.array(depth)
    cR = contrast(rgb[:,:,0],n_size)
    cG = contrast(rgb[:,:,1],n_size)
    cB = contrast(rgb[:,:,2],n_size)
    cD = contrast(depth,11)
    diss = central_mapp(depth)
    # print(diss.shape,cD.shape,(cR+cG+cB).shape)
    return normm(otsu*diss),normm(cD*diss),normm((cR+cG+cB)*diss)    
    
class TrDataset(torch.utils.data.Dataset):
    def __init__(self, root, trainsize, extra_scales=None):
        super().__init__()
        if extra_scales is not None:
            self.scales = (1,) + tuple(extra_scales)
        self.image_dir = os.path.join(root,"RGB")
        self.depth_dir = os.path.join(root,"depth")
        self.mask_dir = os.path.join(root,"GT")
        self.image_paths = os.listdir(self.image_dir)
        self.depth_paths = os.listdir(self.depth_dir)
        self.mask_paths = os.listdir(self.mask_dir)
        
        self.joint_trans = A.Compose(
            [
                A.Resize(height=trainsize, width=trainsize),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=90),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.75),
                A.Normalize(),
            ],
            additional_targets=dict(depth="mask",s1="mask",s2="mask",s3="mask"),  # For RGBD dataset
        )    
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.image_paths[index])
        depth_path = os.path.join(self.depth_dir,self.depth_paths[index])
        mask_path = os.path.join(self.mask_dir,self.mask_paths[index])
        # edge_path = os.path.join(self.edge_dir,self.edge_paths[index])
        # print(image_path,depth_path,mask_path)
        image = read_color_array(image_path)
        mask = read_binary_array(mask_path, to_normalize=True, thr=0.5)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)
        otsu = otsu_map(depth,2)
        s1,s2,s3=saliency_priors(image,depth,otsu)
        transformed = self.joint_trans(image=image, mask=mask, depth=depth,s1=s1,s2=s2,s3=s3)
        image = transformed["image"]
        mask = transformed["mask"]
        depth = transformed["depth"]
        s1 = transformed["s1"]
        s2 = transformed["s2"]
        s3 = transformed["s3"]
        h , w = depth.shape
        bin = np.zeros((3,h,w))
        bin[0] = s1
        bin[1] = s2
        bin[2] = s3
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        bin_tensor = torch.from_numpy(bin)
        return dict(image=image_tensor,mask=mask_tensor,depth=depth_tensor,bin=bin_tensor)

#dataloader for training
def get_loader(tr_data_paths, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):
    dataset = TrDataset(tr_data_paths,trainsize)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  pin_memory=pin_memory)
    return data_loader     


class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root, trainsize):
        super().__init__()
        self.image_dir = os.path.join(root,"RGB")
        self.depth_dir = os.path.join(root,"depth")
        self.image_paths = os.listdir(self.image_dir)
        self.depth_paths = os.listdir(self.depth_dir)
        self.joint_trans = A.Compose([A.Resize(height=trainsize, width=trainsize), A.Normalize()],
            additional_targets=dict(s1="mask",s2="mask",s3="mask"),  # For RGBD dataset
        )
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir,self.image_paths[index])
        depth_path = os.path.join(self.depth_dir,self.depth_paths[index])
        image = read_color_array(image_path)
        depth = read_binary_array(depth_path, to_normalize=True, thr=-1)
        otsu = otsu_map(depth,2)
        s1,s2,s3=saliency_priors(image,depth,otsu)
        transformed = self.joint_trans(image=image, mask=depth,s1=s1,s2=s2,s3=s3)
        image = transformed["image"]
        depth = transformed["mask"]
        s1 = transformed["s1"]
        s2 = transformed["s2"]
        s3 = transformed["s3"]
        h , w = depth.shape
        bin = np.zeros((3,h,w))
        bin[0] = s1
        bin[1] = s2
        bin[2] = s3
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        bin_tensor = torch.from_numpy(bin)
        return dict(
            image=image_tensor,
            depth=depth_tensor,
            bin=bin_tensor,
            name=self.depth_paths[index],
            h=h,
            w=w,
        )    