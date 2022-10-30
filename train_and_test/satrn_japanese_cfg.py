
train = dict(
    type='OCRDataset',
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='train_labels.json',
    test_mode=False,
    pipeline=None)

test = dict(
    type='OCRDataset',
    data_prefix=dict(img_path='mnt/ramdisk/max/90kDICT32px'),
    ann_file='test_labels.json',
    test_mode=True,
    pipeline=None)

default_scope = 'mmocr'

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

randomness = dict(seed=None)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    visualization=dict(
        type='VisualizationHook',
        interval=1,
        enable=False,
        show=False,
        draw_gt=False,
        draw_pred=False))

log_level = 'INFO'
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
load_from = None
resume = False

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ])

test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(
            type='WordMetric',
            mode=['exact', 'ignore_case', 'ignore_case_symbol']),
        dict(type='CharMetric')
    ])

vis_backends = [dict(type='LocalVisBackend')]

visualizer = dict(
    type='TextRecogLocalVisualizer',
    name='visualizer',
    vis_backends=[dict(type='LocalVisBackend')])

optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adam', lr=0.0003))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [dict(type='MultiStepLR', milestones=[3, 4], end=5)]

file_client_args = dict(backend='disk')

dictionary = dict(
    type='Dictionary',
    dict_file=
    'd:/python/preocrenv/lib/site-packages/mmocr/.mim/configs/textrecog/satrn/../../../dicts/english_digits_symbols.txt',
    with_padding=True,
    with_unknown=True,
    same_start_end=True,
    with_start=True,
    with_end=True)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SATRNEncoder',
        n_layers=12,
        n_head=8,
        d_k=64,
        d_v=64,
        d_model=512,
        n_position=200,
        d_inner=2048,
        dropout=0.1),
    decoder=dict(
        type='NRTRDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=2048,
        d_k=64,
        d_v=64,
        module_loss=dict(
            type='CEModuleLoss', flatten=True, ignore_first_char=True),
        dictionary=dict(
            type='Dictionary',
            dict_file=
            'd:/python/preocrenv/lib/site-packages/mmocr/.mim/configs/textrecog/satrn/../../../dicts/english_digits_symbols.txt',
            with_padding=True,
            with_unknown=True,
            same_start_end=True,
            with_start=True,
            with_end=True),
        max_seq_len=25,
        postprocessor=dict(type='AttentionPostprocessor')),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=dict(backend='disk'),
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(200, 32), keep_ratio=False),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='RandomRotate',
                        max_angle=5,
                    ),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(type='PyramidRescale'),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='GaussNoise', var_limit=(20, 20), p=0.5),
                    dict(type='MotionBlur', blur_limit=5, p=0.5),
                ]),
        ]),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='Resize', scale=(200, 32), keep_ratio=False),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

train_dataset = dict(
    type='ConcatDataset',
    datasets=[train],
    pipeline=[
        dict(
            type='LoadImageFromFile',
            file_client_args=dict(backend='disk'),
            ignore_empty=True,
            min_size=2),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(type='Resize', scale=(200, 32), keep_ratio=False),
        dict(
            type='PackTextRecogInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    ])

test_dataset = dict(
    type='ConcatDataset',
    datasets=[test],
    pipeline=[
        dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
        dict(type='Resize', scale=(200, 32), keep_ratio=False),
        dict(type='LoadOCRAnnotations', with_text=True),
        dict(
            type='PackTextRecogInputs',
            meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
    ])

train_dataloader = dict(
    batch_size=128,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=test_dataset)

auto_scale_lr = dict(base_batch_size=512)
