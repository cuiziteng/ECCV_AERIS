_base_ = '../centernet_up_stage2.py'

model = dict(
    type='CenterNet_Up_Re',
    scale_cfg = dict(scale_list=[12, 13, 14, 15, 16]),)