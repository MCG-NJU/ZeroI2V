_base_ = [
    '../../_base_/default_runtime_on_ceph_hdd.py'
]

# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ViT_Zero_CLIP_ablation',
        pretrained="pretrained_models",
        input_resolution=224,
        patch_size=16,
        num_frames=32,
        width=768,
        layers=12,
        heads=12,
        adapter_scale=0.5,
        num_tadapter=2,
        stdha_cfg=dict(shift_div=12, divide_head=False, long_shift_div=12, long_shift_right=True)
        ),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=51,
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
        data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[122.769, 116.74, 104.04],
        std=[68.493, 66.63, 70.321], # 注意clip和imagenet的不一样
        # blending=dict(
        #     type='RandomBatchAugment',
        #     augments=[
        #         dict(type='MixupBlending', alpha=0.8, num_classes=400),
        #         dict(type='CutmixBlending', alpha=1, num_classes=400)
        #     ]),
        format_shape='NCTHW'))


# dataset settings
split = 1
dataset_type = 'RawframeDataset'
data_root = 'your_path'
data_root_val = data_root
ann_file_train = f'your_path/hmdb51/hmdb51_train_split_{split}_rawframes.txt'
ann_file_val = f'your_path/hmdb51/hmdb51_val_split_{split}_rawframes.txt'
ann_file_test = f'your_path/hmdb51/hmdb51_val_split_{split}_rawframes.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='UniformSample', clip_len=32),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='PytorchVideoWrapper',
        op='RandAugment',
        magnitude=7,
        num_layers=4),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    # dict(type='RandomErasing', erase_prob=0.25, mode='rand'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='UniformSample', clip_len=32, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='UniformSample', clip_len=32, num_clips=2, test_mode=True),
    dict(type='RawFrameDecode', **file_client_args),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(img=data_root),
        filename_tmpl='img_{:05}.jpg',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(img=data_root),
        filename_tmpl='img_{:05}.jpg',
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
        data_prefix=dict(img=data_root),
        filename_tmpl='img_{:05}.jpg',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=40, val_begin=1, dynamic_intervals=[(1, 5), (20, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# optimizer
optim_wrapper = dict(
    type='GradMonitorAmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05),
    constructor='GradMonitorSwinOptimWrapperConstructor',
    paramwise_cfg=dict(class_embedding=dict(decay_mult=0.),
                        positional_embedding=dict(decay_mult=0.),
                        temporal_embedding=dict(decay_mult=0.),
                        absolute_pos_embed=dict(decay_mult=0.),
                        ln_1=dict(decay_mult=0.),
                        ln_2=dict(decay_mult=0.),
                        ln_pre=dict(decay_mult=0.),
                        ln_post=dict(decay_mult=0.)
                                    ))

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
auto_scale_lr = dict(enable=False, base_batch_size=64)

