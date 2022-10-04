_base_ = '../centernet_up_stage2.py'

model = dict(
    type='CenterNet_Up_Re',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=False,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='CTNeck_SR_Dual',
        in_channel=512,
        sr_channel=32),
    arrd = dict(type='Adaptive_SR', in_c=32),)