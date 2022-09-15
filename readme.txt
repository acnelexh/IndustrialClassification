================================================================================================================
训练：
./tools/dist_train.sh 训练参数.py #gpu数量

例如
./tools/dist_train.sh ./models_industrial/best_insulator.py 2
用best_insulator.py的参数，2个gpu来训练

================================================================================================================

测试：
./tools/dist_test.sh 训练参数.py 模型储存路径 #gpu数量 --out 结果.csv --metrics 指标 --metric-options 指标参数

例如
./tools/dist_test.sh best_insulator.py ./work_dirs/resnet50_insulator/epoch_88.pth 2 --out tmp.json --metrics accuracy --metric-options topk=1
用best_insulator.py的参数训练的第88个模型，2个gpu，把结果保存在tmp.json，top 1 准确率

================================================================================================================

画图：
./plot_curve.sh

在plot_curve.sh里面改以下的参数

exp="20220831_132921" #实验名字
JSON_LOGS="./work_dirs/resnet50_wire/${exp}.log.json" #实验结果储存路径
KEYS="loss" #需要画loss
TITLE="loss_plot" #画图标题
OUT_FILE="loss_plot_${exp}.png" #储存名字

KEYS="accuracy_top-1" #画acc
TITLE="acc_plot" #画图标题
OUT_FILE="acc_plot_${exp}.png" #储存名字

================================================================================================================

新数据集：
如果有新的数据集，可以使用自带的custom dataset class来读入
只需要把数据集按照类似一下构造就可以
├── test
│   ├── electric
│   └── normal
├── train
│   ├── electric
│   └── normal
├── val
│   ├── electric
└── └── normal
然后在训练参数.py里面注明数据集的位置，类似以下的data_predix，然后就会自动读取数据了
也需要标注不同的标签
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

================================================================================================================

训练参数.py

例子
_base_ = [
    './configs/_base_/models/resnet50.py',
    './configs/_base_/datasets/wire.py']

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
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50, 70])
runner = dict(type='EpochBasedRunner', max_epochs=100)

checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]

其中_base_里面的文件时inherited下来的参数
