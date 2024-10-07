_base_ = ['./configs/detr/detr_r50_8xb2-150e_coco.py']

dataset_type = 'CocoDataset'
classes = ('Person','Ear','Earmuffs','Face','Face-guard','Face-mask-medical','Foot','Tools',
    'Glasses','Gloves','Helmet','Hands','Head','Medical-suit','Shoes','Safety-suit','Safety-vest')
data_root='data'

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='train100.json',
        data_prefix=dict(img='datasets/train/images100')
        )
    )

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='valid100.json',
        data_prefix=dict(img='datasets/valid/images100')
        )
    )

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        # explicitly add your class names to the field `metainfo`
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file='test/annotation_data',
        data_prefix=dict(img='datasets/test/images100')
        )
    )
model = dict(
    bbox_head=dict(
        num_classes=17))

val_evaluator = dict(  # Validation evaluator config
    type='CocoMetric',  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + '/valid100.json',  # Annotation file path
    metric=['bbox'],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator  # Testing evaluator config


# optimizer
# lr is set for a batch size of 8
#optim_wrapper = dict(optimizer=dict(lr=0.01))


# max_epochs
train_cfg = dict(max_epochs=8)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])

load_from = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'