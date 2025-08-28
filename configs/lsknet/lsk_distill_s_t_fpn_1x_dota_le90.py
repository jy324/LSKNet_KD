_base_ = './lsk_t_fpn_1x_dota_le90.py'
from copy import deepcopy  # noqa: F401

angle_version = 'le90'

# Teacher (large) config (structure only; weights via teacher_ckpt)
teacher_ckpt = 'checkpoints/lsk_s_fpn_1x_dota_le90.pth'  # TODO: replace path
teacher_model = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2, 2, 4, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0] * 6,
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=15,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
)

# After inheriting base, `model` refers to student (tiny) model.
student = deepcopy(model)
student['roi_head']['bbox_head']['loss_distill'] = dict(type='KnowledgeDistillationKLDivLoss', T=2.0, loss_weight=2.0)

model = dict(type='KDOrientedRCNN', teacher_config=teacher_model, teacher_ckpt=teacher_ckpt, student=student)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00015, betas=(0.9, 0.999), weight_decay=0.05)

find_unused_parameters = False  # teacher has no grads; all student params used