# Exploring Resolution and Degradation Clues as Self-supervised Signal for Low Quality Object Detection


## Abstract:
Image restoration algorithms such as super resolution (SR) are indispensable pre-processing modules for object detection in low quality images. Most of these algorithms assume the degradation is fixed and known a priori. However, in practical, either the real degradation or optimal up-sampling ratio rate is unknown or differs from assumption, leading to a deteriorating performance for both the pre-processing module and the consequent high-level task such as object detection. Here, we propose a novel self-supervised framework to detect objects in degraded low resolution images. We utilizes the downsampling degradation as a kind of transformation for self-supervised signals to explore the equivariant representation against various resolutions and other degradation conditions. The Auto Encoding Resolution in Self-supervision (AERIS) framework could further take the advantage of advanced SR architec- tures with an arbitrary resolution restoring decoder to reconstruct the original correspondence from the degraded input image. Both the rep- resentation learning and object detection are optimized jointly in an end-to-end training fashion. The generic AERIS framework could be implemented on various mainstream object detection architectures with different backbones. The extensive experiments show that our methods has achieved superior performance compared with existing methods when facing variant degradation situations.


<div align="center">
  <img src="./pics/demo.png" height="300">
</div>
<p align="center">
  Fig 1. The gap between recognition methods and restoration methods.
</p>


<div align="center">
  <img src="./pics/demo.png" height="300">
</div>
<p align="center">
  Fig 2. The scale gap in object detection task.
</p>


<div align="center">
  <img src="./pics/fig1.png" height="400">
</div>
<p align="center">
  Fig 3. We use AERIS to find the scale-invariant feature for object detection under low resolution.
</p>

## Environment:
The code is base on mmdet 2.23.0, to use this code please following the steps build enviroment:

```
conda create -n AERIS python==3.7.0

conda activate AERIS

conda install --yes -c pytorch pytorch=1.7.0 torchvision cudatoolkit=11.0 (for other version, please follow [pytorch](https://pytorch.org/))



pip install opencv-python scipy

pip install -r requirements.txt

pip install -v -e .
```



## Dataset:

**COCO-d dataset:** Translate COCO-val2017 with random noise/ blur/ down-sampling(1~4) resolution, details please refer to our paper.

Download the COCO-d dataset. Dataset link: [(g-drive)](https://drive.google.com/file/d/1mXh_6KZm1ujgc1UOsOPXtPV4dKS_hx9F/view?usp=sharing) or [(baiduyun, passwd:p604)](https://pan.baidu.com/s/1HBo74HpRqIOT0m4_tzKlTg). Label link: [g-drive](https://drive.google.com/file/d/1blrN-QpGmSVyu26Ts5xnDmsAQBcuDtiO/view?usp=sharing) or [(baiduyun, passwd:iosl)](https://pan.baidu.com/s/1VmEnDr8MRiAx_Pca5GkIow).

**Specific Degradation Setting:** 
TBD

**Generate Degradation Dataset by Your Own**: 
TBD

## Model inference:

**1. Download the Pre-train weights:** 



## Model Training:

AERIS on stage 2:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh configs/centernet_AERIS/centernet_up_stage2.py 4
```

AERIS on stage 1:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 bash tools/dist_train.sh configs/centernet_AERIS/centernet_up_stage1.py 4
```

## Citation:

The work is heavily build on [mmdetection](https://github.com/open-mmlab/mmdetection) and [kornia](https://openaccess.thecvf.com/content_WACV_2020/html/Riba_Kornia_an_Open_Source_Differentiable_Computer_Vision_Library_for_PyTorch_WACV_2020_paper.html), thanks for their great project!

If you find this work useful in your research, please consider to cite:

```
@misc{ECCV_AERIS,
  doi = {10.48550/ARXIV.2208.03062},
  url = {https://arxiv.org/abs/2208.03062},
  author = {Cui, Ziteng and Zhu, Yingying and Gu, Lin and Qi, Guo-Jun and Li, Xiaoxiao and Zhang, Renrui and Zhang, Zenghui and Harada, Tatsuya},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Exploring Resolution and Degradation Clues as Self-supervised Signal for Low Quality Object Detection},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```




