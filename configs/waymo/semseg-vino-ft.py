_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16
num_worker = 192
batch_size_val = None
batch_size_test = None
clip_grad = None
enable_amp = True
mix_prob = 0.4


# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=22,
    backbone_out_channels=1024,
    backbone=dict(
        type="Volt",
        in_channels=7,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1e-5,
        qk_norm=True,
        drop_path=0.1,
        increase_drop_path=True,
        up_mlp_dim=256,
        num_register_tokens=16,
        tokenizer_mean_pool=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            label_smoothing=0.1,
        ),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=False,
)

# scheduler settings
epoch = 3
eval_epoch = 1
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.1)
param_dicts = [
    dict(keyword="backbone.tokenizer", lr=0.00005),
    dict(keyword="backbone.register_tokens", lr=0.00005),
    dict(keyword="backbone.blocks", lr=0.00005),
    dict(keyword="backbone.norm", lr=0.00005),
]
scheduler = dict(
    type="OneCycleLR",
    max_lr=[],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
dataset_type = "WaymoDataset"
data_root = "data/waymo"

data = dict(
    num_classes=22,
    ignore_index=-1,
    names=[
        "Car",
        "Truck",
        "Bus",
        "Other Vehicle",
        "Motorcyclist",
        "Bicyclist",
        "Pedestrian",
        "Sign",
        "Traffic Light",
        "Pole",
        "Construction Cone",
        "Bicycle",
        "Motorcycle",
        "Building",
        "Vegetation",
        "Tree Trunk",
        "Curb",
        "Road",
        "Lane Marker",
        "Other Ground",
        "Walkable",
        "Sidewalk",
    ],
    train=dict(
        type=dataset_type,
        split="training",
        data_root=data_root,
        ignore_index=-1,
        transform=[
            dict(type="EstimateNormal"),
            dict(type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.0025, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="TokenCrop", max_token_size=15000, token_stride=5),
            dict(type="NormalToProj"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("strength", "proj"),
            ),
        ],
        test_mode=False,
        loop=1,
    ),
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        ignore_index=-1,
        transform=[
            dict(type="EstimateNormal"),
            dict(type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="NormalToProj"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("strength", "proj"),
            ),
        ],
        test_mode=False,
        loop=1,
    ),
    test=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        transform=[
            dict(type="EstimateNormal"),
            dict(type="PointClipDistance", max_dist=75.0, z_min=-4.0, z_max=2.0),
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="NormalToProj"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("strength", "proj"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=-1,
    ),
)

weight = None
hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
