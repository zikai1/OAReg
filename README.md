<p align="center">
  <p align="center">
    <h1 align="center">OAReg (ICLR 2025)</h1>
  </p>
  <p align="center" style="font-size:16px">
    <a target="_blank" href="https://zikai1.github.io/"><strong>Mingyang Zhao</strong></a>
    ·
    <a target="_blank" href="https://scholar.google.com/citations?user=5hti_r0AAAAJ"><strong>Gaofeng Meng</strong></a>
   ·
    <a target="_blank" href="https://sites.google.com/site/yandongming/"><strong>Dong-Ming Yan</strong></a>
  </p>

![](./fig/ICLR_Teaser.png)
### [Project Page](https://zikai1.github.io/pub/CluReg/index.html) | [Paper](https://arxiv.org/abs/2406.18817) | [Poster](https://zikai1.github.io/slides/CVPR24_Creg_poster.pdf)
This repository contains the official implementation of our ICLR 2025 paper "Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy". 

**Towards Non-rigid or Deformable Registration of Point Clouds and Surfaces**

**Please give a star if you find this repo useful 🤡**


## Implementation
### 1. Prerequisites ###
The code is based on PyTorch implementation, and tested on the following environment dependencies:
```
- Linux (tested on Ubuntu 22.04.1)
- Python 3.9.19
- torch=='1.12.1+cu113'
```

### 2. Setup ###
We recommend using ```Miniconda``` to set up the environment. 

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


If you want the torch version match the pytorch3d version, please use ```conda list``` to check the corresponding Version, and then re-setup the torch, such as
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

 The deformed point clouds are save in the subdirectory ```save_deformed``` of the directory ```data``` .
 

## Contact 
If you have any problem, please contact us via <migyangz@gmail.com>. We greatly appreciate everyone's feedback and insights. Please do not hesitate to get in touch!

## Citation
Please give a citation of our work if you find it useful:

```bibtex
@inproceedings{zhao2025oareg,
  title={Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy},
  author={Mingyang Zhao, Gaofeng Meng, Dong-Ming Yan},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
## Acknowledgements
Our work is inspired by several outstanding prior works, including [DPF](https://github.com/sergeyprokudin/dpf), [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [NDP](https://github.com/rabbityl/DeformationPyramid), and others. We would like to acknowledge and express our deep appreciation to the authors of these remarkable contributions.


## License
OAReg is under AGPL-3.0, so any downstream solution and products (including cloud services) that include OAReg code inside it should be open-sourced to comply with the AGPL conditions. For learning purposes only and not for commercial use. If you want to use it for commercial purposes, please contact us first.




