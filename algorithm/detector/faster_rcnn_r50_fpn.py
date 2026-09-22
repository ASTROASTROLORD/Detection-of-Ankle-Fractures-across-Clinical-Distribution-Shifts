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
    roi_head=dict(bbox_head=dict(num_classes=len(class_names))),
    test_cfg=dict(rcnn=dict(score_thr=0.05, max_per_img=100)),
)
