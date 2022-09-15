model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=1))
dataset_type = 'CustomDataset'
classes = ['normal', 'electric']
train_pipeline = [
    dict(type='Resize', size=128),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=128),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_prefix='/home/disk1/czy/data/insulator_cls_split3/train',
        classes=['normal', 'electric'],
        pipeline=[
            dict(type='Resize', size=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CustomDataset',
        data_prefix='/home/disk1/czy/data/insulator_cls_split3/val',
        classes=['normal', 'electric'],
        pipeline=[
            dict(type='Resize', size=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CustomDataset',
        data_prefix='/home/disk1/czy/data/insulator_cls_split3/test',
        classes=['normal', 'electric'],
        pipeline=[
            dict(type='Resize', size=128),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
train_cfg = dict(mixup=dict(alpha=0.2, num_classes=2))
evaluation = dict(
    interval=1, metric='accuracy', metric_options=dict(topk=(1, )))
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[50, 70])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]
work_dir = './work_dirs/resnet50_insulator'
gpu_ids = range(0, 2)