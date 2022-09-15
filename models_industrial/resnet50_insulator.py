_base_ = [
    './configs/_base_/models/resnet50.py',
    './configs/_base_/datasets/insulator.py']

model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(
        num_classes=2,
        topk=(1)),
)

train_cfg = dict(mixup=dict(alpha=0.2, num_classes=2))

evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, )})

# optimizer
optimizer = dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[160])
runner = dict(type='EpochBasedRunner', max_epochs=200)

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]

