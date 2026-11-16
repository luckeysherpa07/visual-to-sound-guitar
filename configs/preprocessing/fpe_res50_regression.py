# GuiFiR fretboard pose estimator (FPE): ResNet-50 -> global average pooling ->
# linear regression of 126 (x, y) grid points, weighted smooth L1 loss.
# Paper: GuiFiR Sec. IV-A.2. Pretraining hyperparameters are not given in the
# paper; optimiser/schedule follow MMPose's regression defaults.
# Run with scripts/01d_train_fpe.py (sets work_dir under outputs/runs/).
_base_ = ['mmpose::_base_/default_runtime.py']

data_root = 'data/'
ann_dir = 'processed/fretboard_coco/'  # relative to data_root
metainfo = dict(from_file='configs/preprocessing/fretboard_126_metainfo.py')

# runtime
train_cfg = dict(max_epochs=100, val_interval=5)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=100, milestones=[70, 90], gamma=0.1,
         by_epoch=True),
]

auto_scale_lr = dict(base_batch_size=32)

default_hooks = dict(checkpoint=dict(save_best='EPE', rule='less', max_keep_ckpts=2))

# codec: normalised (x, y) in the 256x256 crop
codec = dict(type='RegressionLabel', input_size=(256, 256))

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RegressionHead',
        in_channels=2048,
        num_joints=126,
        loss=dict(type='SmoothL1Loss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=False))

dataset_type = 'CocoDataset'
data_mode = 'topdown'

# COCO bboxes are tight around the 126 points; padding=1.1 gives the paper's 10%.
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.1),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', scale_factor=(0.8, 1.2), rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PhotometricDistortion'),
    dict(type='GenerateTarget', encoder=codec, use_dataset_keypoint_weights=True),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.1),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]


def _dataset(ann, pipeline, test_mode=False):
    return dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=ann_dir + ann,
        data_prefix=dict(img=''),
        metainfo=metainfo,
        pipeline=pipeline,
        test_mode=test_mode)


train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=_dataset('train.json', train_pipeline))
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=_dataset('val.json', val_pipeline, test_mode=True))
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=_dataset('test.json', val_pipeline, test_mode=True))

# EPE: mean point error in pixels; NME normalised by nut/high-E <-> f20/low-E diagonal.
val_evaluator = [
    dict(type='EPE'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 125]),
]
test_evaluator = val_evaluator
