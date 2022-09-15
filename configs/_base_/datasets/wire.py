# dataset settings
dataset_type = 'CustomDataset'

classes = ['normal', 'broken']

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
        type=dataset_type,
        data_prefix='/home/disk1/czy/data/wire_balance/train',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='/home/disk1/czy/data/wire_balance/val',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_prefix='/home/disk1/czy/data/wire_balance/test',
        classes=classes,
        pipeline=test_pipeline,
        test_mode=True))
