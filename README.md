# Coding

![](demo_video.gif)

## Dependencies
- Python 3.10
- PyTorch 2.4.1
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
- `conda create -n Coding python=3.10`
- `conda activate Coding`
- `pip install -r requirements.txt`

## Testing

- Download seven test datasets from [GoogleDrive](https://drive.google.com/drive/folders/1xgq6xN7L1wAfUcR-bJuue_VqJ81J2d52?usp=sharing) and keep in location 
`dataset/TestDataset`

- Download checkpoint from [GoogleDrive](https://drive.google.com/drive/folders/19nnI3-COJpvlXPg-y7LfnLx4ZdvA8eXk?usp=sharing) and keep in location 
`ckpt`

- Generate the saliency maps
`python test.py`

- Test the four evaluation metrics on seven test datasets
`python test_evaluation_maps.py`


- The saliency maps are in `saliency maps/`.
 
- Our saliency maps can also be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1-MxTy_3Yn5Ji-cYeK9mElmlTtSHpHGlz?usp=sharing).
