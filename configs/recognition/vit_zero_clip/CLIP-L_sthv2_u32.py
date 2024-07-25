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
resume = True
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_Zero_CLIP_long',
        pretrained='your_clip_path',
        input_resolution=224,
        patch_size=14,
        num_frames=32,
        width=1024,
        layers=24,
        heads=16,
        adapter_scale=1,
        num_tadapter=2,
        stdha_cfg=dict(
            divide_head=False,
            shift1_div=8,
            shift2_div=16,
            shift3_div=16,
            shift3_right=False,
            shift4_div=-1,
            shift4_right=False)),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
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
dataset_type = 'VideoDataset'
data_prefix = 'your_path_sthv2/'
data_root = 'your_path_sthv2/videos'
data_root_val = 'your_path_sthv2/videos'
ann_file_train = 'data/sthv2/sthv2_train_list_videos.txt'
ann_file_val = 'data/sthv2/sthv2_val_list_videos.txt'
ann_file_test = 'data/sthv2/sthv2_val_list_videos.txt'
file_client_args = dict(io_backend='ceph')
sthv2_flip_label_map = dict({
    86: 87,
    87: 86,
    93: 94,
    94: 93,
    166: 167,
    167: 166
})
train_pipeline = [
    dict(type='DecordInit', io_backend='ceph'),
    dict(type='UniformSample', clip_len=32),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',
        flip_ratio=0.5,
        flip_label_map=dict({
            86: 87,
            87: 86,
            93: 94,
            94: 93,
            166: 167,
            167: 166
        })),
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
    dict(type='DecordInit', io_backend='ceph'),
    dict(type='UniformSample', clip_len=32, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', io_backend='ceph'),
    dict(type='UniformSample', clip_len=32, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/sthv2/sthv2_train_list_videos.txt',
        data_prefix=dict(video='your_path_sthv2/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(type='UniformSample', clip_len=32),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='RandomResizedCrop'),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(
                type='Flip',
                flip_ratio=0.5,
                flip_label_map=dict({
                    86: 87,
                    87: 86,
                    93: 94,
                    94: 93,
                    166: 167,
                    167: 166
                })),
            dict(
                type='PytorchVideoWrapper',
                op='RandAugment',
                magnitude=7,
                num_layers=4),
            dict(type='RandomErasing', erase_prob=0.25),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ]))
val_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VideoDataset',
        ann_file='data/sthv2/sthv2_val_list_videos.txt',
        data_prefix=dict(video='your_path_sthv2/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(type='UniformSample', clip_len=32, test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
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
        ann_file='data/sthv2/sthv2_val_list_videos.txt',
        data_prefix=dict(video='your_path_sthv2/videos'),
        pipeline=[
            dict(type='DecordInit', io_backend='ceph'),
            dict(type='UniformSample', clip_len=32, test_mode=True),
            dict(type='DecordDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='ThreeCrop', crop_size=224),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='PackActionInputs')
        ],
        test_mode=True))
val_evaluator = dict(type='AccMetric')
test_evaluator = dict(type='AccMetric')
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_begin=1,
    dynamic_intervals=[(1, 5), (40, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
accumulative_counts = 2
auto_scale_lr = dict(enable=True, base_batch_size=32)
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0003, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    accumulative_counts=2,
    paramwise_cfg=dict(
        class_embedding=dict(decay_mult=0.0),
        positional_embedding=dict(decay_mult=0.0),
        temporal_embedding=dict(decay_mult=0.0),
        absolute_pos_embed=dict(decay_mult=0.0),
        ln_1=dict(decay_mult=0.0),
        ln_2=dict(decay_mult=0.0),
        ln_pre=dict(decay_mult=0.0),
        ln_post=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1)))
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
launcher = 'slurm'
work_dir = '.work_dirs/recognition/vit_zero_clip/FinalS_vit_zero_large_clip_sthv2_u32_adapter_2'
randomness = dict(seed=None, diff_rank_seed=False, deterministic=False)
dist_params = dict(port=18757)
