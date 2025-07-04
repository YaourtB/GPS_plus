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
(Optional) For training with geometry regulatization, install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for **chamfer_distance**. Otherwise, set ```if_chamfer = False``` in [train.py](train.py#L69).  

## Data Preparation

1. You can download our captured THumanMV dataset from [OneDrive](https://docs.google.com/forms/d/e/1FAIpQLSexKlYfpUFcgnKM7EYoIFWi7P3J1InlHyTC82ehqka2hTiwmA/viewform?usp=dialog). We provide 15 sequences of human performance captured in 10-camera setting. In our experiments, we split 10 cameras into 3 work sets: (1,2,3,4) (4,5,6,7) (7,8,9,10).

2. We provide [step_0rect.py](data_process/step_0rect.py) for source view rectification and [step_1.py](data_process/step_1.py) for novel view processing. To prepare data, you set the correct path for ```data_root```(raw data) and ```processed_data_root```(processed data) in [step_0rect.py](data_process/step_0rect.py#L99) and [step_1.py](data_process/step_1.py#L13). Then you can run for example:
```
cd data_process
python step_0rect.py -i s1a1 -t train
python step_1.py -i s1a1 -t train

python step_0rect.py -i s3a5 -t val
python step_1.py -i s3a5 -t val

python step_0rect.py -i s1a6 -t test
python step_1.py -i s1a6 -t test
cd ..
```

The processed dataset should be organized as follows: 
```
processed_data_root
├── train/
│   ├── img/
│   │   ├── s1a1_s1_0000/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   │   │   ├── 2.jpg
│   │   │   ├── 3.jpg
│   │   │   ├── 4.jpg
│   │   |   └── 5.jpg
│   |   └── ...
│   ├── mask/
│   │   ├── s1a1_s1_0000/
│   │   │   ├── 0.jpg
│   │   │   ├── 1.jpg
│   |   └── ...
│   ├── parameter/
│   │   ├── s1a1_s1_0000/
│   │   │   ├── 0_1.json
│   │   │   ├── 2_extrinsic.npy
│   │   │   ├── 2_intrinsic.npy
│   |   |   └── ...
│   |   └── ...
└──val
│   ├── img/
│   ├── mask/
│   ├── parameter/
└──test
│   ├── s1a6_process/
│   |   ├── img/
│   |   ├── mask/
│   |   ├── parameter/
```

Note that 0-1.jpg are rectified input images and 2-5.jpg are images for supervision or evaluation. In particular, 4-5.jpg are original images of 0-1 views.

## Test

We provide the pretrained checkpoint in [OneDrive](https://mailtsinghuaeducn-my.sharepoint.com/:u:/g/personal/bzhou22_mail_tsinghua_edu_cn/ESZX7rn5BLNLtxqcwxW5SGwBA_Z_CBmxx5KoIgtI8lGLPA?e=HvtuTB) and 60-frame processed data in [OneDrive](https://mailtsinghuaeducn-my.sharepoint.com/:u:/g/personal/bzhou22_mail_tsinghua_edu_cn/EXeLFNTNDBxCgkmV3spUMugBjbGEL8QXBL3w7QGOoA7uAw?e=C8VqZa). You can directly put the downloaded data into ```/PATH/TO/processed_data_root/test/```. You furthermore modify ```local_data_root=/PATH/TO/processed_data_root/``` in [stage.yaml](config/stage.yaml#L16)

- For novel-view synthesis, you can set the checkpoint path in [test.py](test.py#L150) and pick a target view in 2-3.
```
python test.py -i example_data -v 2
```

- For freeview rendering, you can set the checkpoint path and ```LOOP_NUM``` in [run_interpolation.py](run_interpolation.py#L268) for frames per work set.
```
python run_interpolation.py -i example_data
```

You can check results in ```experiments\gps_plus```.

## Train

Once you prepare all training data of 9 sequences and at least one sequence as validation data. You can modify ```train_data_root``` and ```val_data_root``` in [stage.yaml](config/stage.yaml#L17).
```
python train.py
```
If you would like to train our network with your own data, you can organize the dataset as above and set ```inverse_depth_init``` in [stage.yaml](config/stage.yaml#L15). We use ```inverse_depth_init = 0.3``` in our experiments for the largest depth of the scene is around 3.33 meters.

## Train on Custom Data

### Custom Data Preparation

1. We assume that you have already calibrated the cameras using the images of the first frame with COLMAP and obtained the "sparse" folder. We provide an example [custom_data.zip](https://github.com/YaourtB/GPS_plus/blob/main/custom_data.zip) in this repo.
2. Organize your sequential custom data into the following structure:
```
raw_custom_data_root
├── sparse/
│   ├── 0/
│   │   ├── cameras.bin
│   │   └── images.bin
├── frame0_cam0.jpg (e.g. 0000_00.jpg)
├── frame0_cam1.jpg
├── ...
├── frame0_camN.jpg
├── ......
├── ......
└── frameT_camN.jpg
```
3. You should determine the size (n) of the work set of cameras, which means how many cameras are there between the leftmost and the rightmost. This number can not be too large. Otherwise, the rectification could collapse. Once you set [```/PATH/TO/custom_data```](https://github.com/YaourtB/GPS_plus/blob/main/data_process/step_0rect_custom.py#L124), you can run followings 
```
cd data_process
python step_0rect_custom.py -t train -n 4
python step_1_custom.py -t train -n 4

python step_0rect_custom.py -t val -n 4
python step_1_custom.py -t val -n 4
```
4. You can probably estimate the ```inverse_depth_init``` in [stage.yaml](config/stage.yaml#L15) by reading the output of the distance of the left-most and the right-most cameras. See [here](https://github.com/YaourtB/GPS_plus/blob/main/data_process/step_0rect_custom.py#L245).
5. Please verify that the correponding pixels of 0.jpg and 1.jpg in the processed data are aligned in the same horizontal line. If it is not the case, the work set size (n) might be too large or the calibration in sparse folder is not good.
6. You can modify ```train_data_root``` and ```val_data_root``` in [stage.yaml](config/stage.yaml#L17) with your [```/PATH/TO/processed_custom_data```](https://github.com/YaourtB/GPS_plus/blob/main/data_process/step_0rect_custom.py#L125) and train our network from scratch.
```
python train.py
```
If the results are not good, you could modify ```inverse_depth_init``` in [stage.yaml](config/stage.yaml#L15) or crop the processed images along with modifying the intrinsic parameters as in [here](https://github.com/YaourtB/GPS_plus/blob/main/data_process/step_0rect.py#L84-L90).

# Citation

If you find the code or the data is useful for your research, please consider citing:
```bibtex
@article{zhou2024gps,
  title={GPS-Gaussian+: Generalizable Pixel-wise 3D Gaussian Splatting for Real-Time Human-Scene Rendering from Sparse Views},
  author={Zhou, Boyao and Zheng, Shunyuan and Tu, Hanzhang and Shao, Ruizhi and Liu, Boning and Zhang, Shengping and Nie, Liqiang and Liu, Yebin},
  journal={arXiv preprint arXiv:2411.11363},
  year={2024}
}
```
