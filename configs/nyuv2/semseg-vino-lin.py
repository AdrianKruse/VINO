_base_ = ["../_base_/default_runtime.py"]

num_worker = 16
batch_size = 16
batch_size_val = None
batch_size_test = None
epoch = 50
eval_epoch = 50
clip_grad = None
enable_amp = True
evaluate = False

model = dict(
    type="DefaultSegmentorV2",
    num_classes=40,
    backbone_out_channels=2304,
    backbone=dict(
        type="Volt",
        in_channels=6,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        init_values=1e-5,
        qk_norm=True,
        drop_path=0.0,
        increase_drop_path=False,
        up_mlp_dim=None,
        num_register_tokens=16,
        tokenizer_mean_pool=True,
        concat_last_n=3,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", label_smoothing=0.1, loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=True,
)

optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.10)
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.001,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

class_names = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "blinds", "desk", "shelves",
    "curtain", "dresser", "pillow", "mirror", "floor mat", "clothes", "ceiling",
    "books", "refridgerator", "television", "paper", "towel", "shower curtain", "box",
    "whiteboard", "person", "night stand", "toilet", "sink", "lamp", "bathtub",
    "bag", "otherstructure", "otherfurniture", "otherprop",
]

data = dict(
    num_classes=40,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type="NYUv2Dataset",
        split="train",
        data_root="data/nyuv2",
        depth_root="data/nyuv2/depth",
        edge_filter=True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="CenterColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "origin_coord", "grid_coord", "segment"),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="NYUv2Dataset",
        split="test",
        data_root="data/nyuv2",
        depth_root="data/nyuv2/depth",
        edge_filter=True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="CenterColor"),
            dict(type="GridSample", grid_size=0.01, hash_type="fnv", mode="train", return_inverse=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))),
                dict(type="Copy", keys_dict={"coord": "origin_coord"}),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "origin_coord", "grid_coord", "index"), feat_keys=("color", "normal")),
            ],
            aug_transform=[[]],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module.backbone."),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]
