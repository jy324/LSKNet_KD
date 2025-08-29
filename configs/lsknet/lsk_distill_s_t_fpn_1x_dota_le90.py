_base_ = ['../_base_/datasets/dotav1.py', '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']

angle_version = 'le90'

# 1. Define teacher model (lsk_s), load its pretrained weights, and set it to not train
teacher_ckpt = 'checkpoints/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth'  # Please replace with your teacher model weight path
teacher_cfg = dict(
    type='OrientedRCNN',
    backbone=dict(
        type='LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2,2,4,2],
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
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
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
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=None
    test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.8),
                min_bbox_size=0),
            rcnn=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(iou_thr=0.1),
                max_per_img=2000))
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 2. Modify student model (lsk_t) config for distillation
model = dict(
    type='KDOrientedRCNN',
    teacher_config=teacher_cfg,
    teacher_ckpt=teacher_ckpt,
    student=dict(  # In KDOrientedRCNN, `model` is renamed to `student`
        type='OrientedRCNN',
        backbone=dict(
            type='LSKNet',
            embed_dims=[32, 64, 160, 256],
            drop_rate=0.1,
            drop_path_rate=0.1,
            depths=[3, 3, 5, 2],
            init_cfg=dict(type='Pretrained', checkpoint="./data/pretrained/lsk_t_backbone.pth"),
            norm_cfg=dict(type='SyncBN', requires_grad=True)),
        neck=dict(
            type='FPN',
            in_channels=[32, 64, 160, 256],
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
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
        roi_head=dict(
            type='OrientedStandardRoIHead',
            bbox_roi_extractor=dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
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
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                # 3. Add distillation loss
                loss_distill=dict(
                    type='KnowledgeDistillationKLDivLoss',
                    loss_weight=2.0, # Distillation loss weight, can be adjusted
                    T=2) # Temperature for softening teacher's output
            )),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    gpu_assign_thr=800,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.8),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    iou_calculator=dict(type='RBboxOverlaps2D'),
                    gpu_assign_thr=800,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.8),
                min_bbox_size=0),
            rcnn=dict(
                nms_pre=2000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(iou_thr=0.1),
                max_per_img=2000))
    )
)

# 4. Adjust optimizer and learning rate
#    Student model might need a smaller learning rate or different training strategy
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001, # Initial learning rate might need to be lowered
    betas=(0.9, 0.999),
    weight_decay=0.05)

# Ensure to find unused parameters, as teacher model's parameters won't be used
find_unused_parameters = True
