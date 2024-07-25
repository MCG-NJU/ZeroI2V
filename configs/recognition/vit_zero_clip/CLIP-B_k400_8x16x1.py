default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_Zero_CLIP',
        pretrained='your_clip_path',
        input_resolution=224,
        patch_size=16,
        num_frames=8,
        width=768,
        layers=12,
        heads=12,
        adapter_scale=0.5,
        num_tadapter=2,
        stdha_cfg=dict(shift_div=12, divide_head=False)),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321],
        format_shape='NCTHW'))
dataset_type = 'VideoDataset'
data_prefix = 'your_path_k400/'
data_root = 'your_path_k400/kinetics_400_320_30fps_train'
data_root_val = 'your_path_k400/kinetics_400_320_30fps_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'
file_client_args = dict(io_backend='ceph')
train_pipeline = [
    dict(type='DecordInit', io_backend='ceph'),
    dict(type='SampleFrames', clip_len=8, frame_interval=16, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', io_backend='ceph'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=16,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', io_backend='ceph'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=16,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/kinetics400/kinetics400_train_list_videos.txt',
        data_prefix=dict(
            video=
            'your_path_k400/kinetics_400_320_30fps_train'
        ),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=16,
                num_clips=1),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='PytorchVideoWrapper',
                op='RandAugment',
                magnitude=7,
                num_layers=4),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/kinetics400/kinetics400_val_list_videos.txt',
        data_prefix=dict(
            video=
            'your_path_k400/kinetics_400_320_30fps_val'
        ),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=16,
                num_clips=1,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/kinetics400/kinetics400_val_list_videos.txt',
        data_prefix=dict(
            video=
            'your_path_k400/kinetics_400_320_30fps_val'
        ),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(
                type='SampleFrames',
                clip_len=8,
                frame_interval=16,
                num_clips=3,
                test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Flip', flip_ratio=0),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_begin=1,
    dynamic_intervals=[(1, 5), (20, 1)])
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
        T_max=40,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=40)
]
find_unused_parameters = True
auto_scale_lr = dict(enable=True, base_batch_size=64)
launcher = 'slurm'
work_dir = '.work_dirs/recognition/vit_zero_clip/N003ddd_vit_zero_base_clip_k400_8x16x1_div12_noerase'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
dist_params = dict(port=30950)
