# OAReg (ICLR 2025)

### Towards Non-rigid or Deformable Registration of Point Clouds and Surfaces

This is an official code for the paper "Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy" (ICLR 2025).

**Please give a star if you find this repo useful.**


## Implementation
### 1. Prerequisites ###
The code is based on PyTorch implementation, and tested on the following environment dependencies:
```
- Linux (tested on Ubuntu 22.04.1)
- Python 3.9.19
- torch=='1.12.1+cu113'
```

### 2. Setup ###
We recommend using Miniconda to set up the environment. 

#### 2.1 Create conda environment #### 
```
- conda create -n oar python=3.9
- conda activate oar
```

#### 2.2 Install packages ####
```
- pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
- conda install -c fvcore -c iopath -c conda-forge fvcore iopath
- conda install pytorch3d
```


If you want the torch version match the pytorch3d version, please use "conda list" to check the corresponding Version, and then re-setup the torch, such as
```
- pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Finally, setup other libraries: 
```
- pip install -r requirements.txt
```

### 3. Test ###
```
- cd src

- python test_OAR.py
```

 The deformed point clouds are save in the subdirectory "save_deformed" of the directory "data" .



