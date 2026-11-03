"""Article-facing Faster R-CNN ResNet-50-FPN detector definition."""

_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
    "mmdet::_base_/default_runtime.py",
]

class_names = (
    "lateral_malleolus_fracture",
    "medial_malleolus_fracture",
    "posterior_malleolus_fracture",
    "distal_tibial_fracture",
)

metainfo = dict(classes=class_names)

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    roi_head=dict(bbox_head=dict(num_classes=len(class_names))),
    test_cfg=dict(rcnn=dict(
        score_thr=0.05, nms=dict(type="nms", iou_threshold=0.50), max_per_img=100,
    )),
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackDetInputs", meta_keys=(
        "img_id", "img_path", "ori_shape", "img_shape", "scale_factor",
    )),
]

# MMDetection's inference API obtains the pipeline from this dataset config.
test_dataloader = dict(dataset=dict(type="CocoDataset", pipeline=test_pipeline))
