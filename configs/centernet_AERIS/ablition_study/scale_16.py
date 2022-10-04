_base_ = '../centernet_up_stage2.py'

model = dict(
    type='CenterNet_Up_Re',
    scale_cfg = dict(scale_list=[16]),)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)