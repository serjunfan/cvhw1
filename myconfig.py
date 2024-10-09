_base_ = ['./configs/detr/detr_r50_8xb2-150e_coco.py']

dataset_type = 'CocoDataset'
classes = ('Person','Ear','Earmuffs','Face','Face-guard','Face-mask-medical','Foot','Tools',
    'Glasses','Gloves','Helmet','Hands','Head','Medical-suit','Shoes','Safety-suit','Safety-vest')
data_root='../dataset'

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train.json',
        data_prefix=dict(img='train/images')
        )
    )

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='valid.json',
        data_prefix=dict(img='valid/images')
        )
    )

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='test/images')
        )
    )
model = dict(
    bbox_head=dict(
        num_classes=17))

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.0001))

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + '/valid.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator  # Testing evaluator config


# optimizer
# lr is set for a batch size of 8
#optim_wrapper = dict(optimizer=dict(lr=0.01))


# max_epochs
train_cfg = dict(max_epochs=100)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))

load_from = './work_dirs1/epoch_45.pth'
resume=True
work_dir = './work_dirs1/'
#load_from = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'