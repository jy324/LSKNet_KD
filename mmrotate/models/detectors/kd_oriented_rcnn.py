from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .two_stage import RotatedTwoStageDetector
import torch
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from .. import builder
import threading

# 全局缓存，避免重复初始化
_teacher_model_cache = {}
_lock = threading.Lock()

@ROTATED_DETECTORS.register_module()
class KDOrientedRCNN(RotatedTwoStageDetector):
    """Knowledge distillation for Oriented R-CNN."""

    def __init__(self,
                 teacher_config,
                 teacher_ckpt,
                 student,
                 train_cfg=None,
                 test_cfg=None):
        super(KDOrientedRCNN, self).__init__(
            backbone=student['backbone'],
            neck=student['neck'],
            rpn_head=student['rpn_head'],
            roi_head=student['roi_head'],
            train_cfg=student['train_cfg'],
            test_cfg=student['test_cfg'],
            init_cfg=getattr(student, "init_cfg", None))

        # 使用缓存的教师模型，避免重复初始化
        cache_key = f"{teacher_ckpt}_{id(teacher_config)}"
        with _lock:
            if cache_key not in _teacher_model_cache:
                print(f"首次初始化教师模型: {teacher_ckpt}")
                _teacher_model_cache[cache_key] = self.build_teacher(teacher_config, teacher_ckpt)
            else:
                print(f"使用缓存的教师模型: {teacher_ckpt}")
            self.teacher_model = _teacher_model_cache[cache_key]
        
        # The student model is the main model in RotatedTwoStageDetector
        self.student_model = self

    def build_teacher(self, config, ckpt):
        """Build teacher model."""
        teacher = builder.build_detector(config)
        load_checkpoint(teacher, ckpt, map_location='cpu')
        teacher.eval()
        # 冻结教师模型参数
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # Teacher forward
        teacher_cls_score = None
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            teacher_proposal_list = self.teacher_model.rpn_head.simple_test_rpn(
                teacher_x, img_metas)
            
            # To get teacher's logits, we need to run roi_head forward
            # We need to assign proposals to gt bboxes for teacher to get features for gt bboxes
            sampling_results = []
            for i in range(len(img_metas)):
                assign_result = self.student_model.roi_head.bbox_assigner.assign(
                    teacher_proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i] if gt_bboxes_ignore else None,
                    gt_labels[i])
                sampling_result = self.student_model.roi_head.bbox_sampler.sample(
                    assign_result,
                    teacher_proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in teacher_x])
                sampling_results.append(sampling_result)

            if len(sampling_results) > 0 and len(sampling_results[0].pos_bboxes) > 0:
                teacher_roi_feats = self.teacher_model.roi_head.bbox_roi_extractor(
                    teacher_x[:self.teacher_model.roi_head.bbox_roi_extractor.num_inputs],
                    torch.cat([res.pos_bboxes for res in sampling_results]))
                teacher_cls_score, _ = self.teacher_model.roi_head.bbox_head(teacher_roi_feats)


        # Student forward
        x = self.student_model.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.student_model.with_rpn:
            proposal_cfg = self.student_model.train_cfg.get('rpn_proposal',
                                            self.student_model.test_cfg.rpn)
            rpn_losses, proposal_list = self.student_model.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.student_model.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 teacher_cls_score=teacher_cls_score,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        return self.student_model.simple_test(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations.

        If specified, this function tests an image with augmentations and then
        merges the results by calling `self.merge_aug_results`.
        """
        return self.student_model.aug_test(imgs, img_metas, **kwargs)

    @property
    def with_rpn(self):
        return self.student_model.with_rpn

    @property
    def with_roi_head(self):
        return self.student_model.with_roi_head
