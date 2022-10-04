## 1. Config File:

### CenterNet-AERIS (ResNet-18 Backbone)

**centernet_res18.py**: config of baseline CenterNet-ResNet model.

**centernet_res18_woARRD.py**: config of baseline CenterNet-ResNet-AERIS without the ARRD decoder.

**centernet_res18_stage1.py**: config of CenterNet-ResNet-AERIS with the ARRD decoder on stage 1.

**centernet_res18_stage2.py**: config of CenterNet-ResNet-AERIS with the ARRD decoder on stage 2.

### CenterNet-AERIS (Swin-Tiny Backbone)

**centernet_swint.py**: config of baseline CenterNet-Swin model.

**centernet_swint_stage2.py**: config of CenterNet-Swin-AERIS with the ARRD decoder on stage 2.


## 2. Model Testing

| Model | Config | Download Link | Training Log | AP | AP small | AP medium | AP large |
|  ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
| CenterNet-Res18 | [config](configs/centernet_AERIS/centernet_res18.py) |  |  | 14.5 | 1.2 | 10.4 | 38.6 | 
| CenterNet-Res18-AERIS (w/o ARRD) | [config](configs/centernet_AERIS/centernet_res18.py) |  |  | 17.6 | 2.5 | 15.9 | 41.4 | 
| CenterNet-Res18-AERIS (Stage1) | [config](configs/centernet_AERIS/centernet_res18_stage1.py) |  |  | 18.2 | 2.6 | 16.2 | 43.2 | 
| CenterNet-Res18-AERIS (Stage2) | [config](configs/centernet_AERIS/centernet_res18_stage2.py) |  |  | 18.4 | 2.7 | 16.4  | 46.5 | 
| CenterNet-SwinTiny | [config](configs/centernet_AERIS/centernet_swint.py) |  |  | 19.9 | 2.7 | 16.9 | 46.2 | 
| CenterNet-SwinTiny-AERIS (Stage2) | [config](configs/centernet_AERIS/centernet_swint_stage2.py) | [model](https://1drv.ms/u/s!AhAeTSGcQBWibIr8OvLkO95pCG4?e=P1Hgbp) |  | 21.6 | 3.2 | 20.4 | 49.0 | 

## 3. Model Training

We default use 4 GPUs for training the model, you can also adjust the learning rate in config:

