<div align="center">

# <b>GPS-Gaussian+</b>: Generalizable Pixel-Wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views

[Boyao Zhou](https://yaourtb.github.io)<sup>1*</sup>, [Shunyuan Zheng](https://shunyuanzheng.github.io)<sup>2*</sup><sup>&dagger;</sup>, [Hanzhang Tu](https://itoshiko.com/)<sup>1</sup>, [Ruizhi Shao](https://dsaurus.github.io/saurus)<sup>1</sup>, [Boning Liu](https://liuboning2.github.io)<sup>1</sup>, [Shengping Zhang](http://homepage.hit.edu.cn/zhangshengping)<sup>2&#x2709;</sup>, [Liqiang Nie](https://liqiangnie.github.io)<sup>2</sup>, [Yebin Liu](https://www.liuyebin.com)<sup>1</sup>

<p><sup>1</sup>Tsinghua Univserity &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<sup>2</sup>Harbin Institute of Technology
 <br>*Equal contribution<sup>&nbsp;&nbsp;&dagger;</sup>Work done during an internship at Tsinghua University&nbsp;&nbsp;<sup>&#x2709</sup>Corresponding author
  
### [Projectpage](https://yaourtb.github.io/GPS-Gaussian+) · [Paper](https://arxiv.org/pdf/2411.11363) · [Dataset](https://docs.google.com/forms/d/e/1FAIpQLSexKlYfpUFcgnKM7EYoIFWi7P3J1InlHyTC82ehqka2hTiwmA/viewform?usp=dialog)
</div>

## Introduction

We present GPS-Gaussian+, a generalizable 3D Gaussian Splatting, for human-centered scene rendering from sparse views in a feed-forward manner.

https://github.com/user-attachments/assets/c24eaf44-ca1f-438a-a538-5a7d5cb78f89


## Installation

To deploy and run GPS-Gaussian+, run the following scripts:
```
conda env create --file environment.yml
conda activate gps_plus
```
Then, compile the ```diff-gaussian-rasterization``` in [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) repository:
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```
(Optional) For training with geometry regulatization, install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for **chamfer_distance**. Otherwise, set *if_chamfer = False* in train.py.  

## Data Preparation

You can download our captured THumanMV dataset from [OneDrive](https://docs.google.com/forms/d/e/1FAIpQLSexKlYfpUFcgnKM7EYoIFWi7P3J1InlHyTC82ehqka2hTiwmA/viewform?usp=dialog). We provide 15 sequences of human performance captured in 10-camera setting. In our experiments, we split 10 cameras into 3 work sets: (1,2,3,4) (4,5,6,7) (7,8,9,10).  
