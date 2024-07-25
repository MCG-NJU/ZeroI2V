_base_ = ['../../_base_/default_runtime_on_local.py']

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_Zero_CLIP_ablation',
        pretrained='your_clip_path',
        input_resolution=224,
        patch_size=16,
        num_frames=16,
        width=768,
        layers=12,
        heads=12,
        dropout_rate=0,
        adapter_scale=1,
        num_tadapter=2,
        stdha_cfg=dict(
            shift_div=12,
            divide_head=False,
            long_shift_div=12,
            long_shift_right=True)),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=174,
        spatial_type='avg',
        dropout_ratio=0.5,
        label_smooth_eps=0.1,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        format_shape='NCTHW'))


# dataset settings
dataset_type = 'VideoDataset'
data_prefix = 'your_path/sthv2/'
data_root = data_prefix+'videos'
data_root_val = data_prefix+'videos'
ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'
ann_file_test = 'data/sthv2/sthv2_val_list_videos.txt'
file_client_args = dict(io_backend='disk')

sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=16),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv2_flip_label_map),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomErasing', erase_prob=0.25),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=16, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSample', clip_len=16, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=1, dynamic_intervals=[(1, 5), (40, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(
        class_embedding=dict(decay_mult=0.0),
        positional_embedding=dict(decay_mult=0.0),
        temporal_embedding=dict(decay_mult=0.0),
        absolute_pos_embed=dict(decay_mult=0.0),
        ln_1=dict(decay_mult=0.0),
        ln_2=dict(decay_mult=0.0),
        ln_pre=dict(decay_mult=0.0),
        ln_post=dict(decay_mult=0.0),
        scale=dict(decay_mult=0.0)))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=2.5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=50)
]

find_unused_parameters = True
auto_scale_lr = dict(enable=False, base_batch_size=64)