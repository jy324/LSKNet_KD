from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .two_stage import RotatedTwoStageDetector
import torch
from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from .. import build_detector
import threading
import torch.distributed as dist
import mmcv
from mmrotate.core import rbbox2roi  # Added for converting bboxes to rois (batch_index + box)

@ROTATED_DETECTORS.register_module()
class KDOrientedRCNN(RotatedTwoStageDetector):
    """Knowledge distillation for Oriented R-CNN."""

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head,
                 teacher_config,
                 output_feature=False,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, rpn_head, roi_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        self.output_feature = output_feature
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')


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

            # Obtain sampling results to extract teacher RoI features aligned with positives
            sampling_results = []
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None] * num_imgs
            for i in range(num_imgs):
                assign_result = self.roi_head.bbox_assigner.assign(
                    teacher_proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i])
                sampling_result = self.roi_head.bbox_sampler.sample(
                    assign_result,
                    teacher_proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in teacher_x])
                sampling_results.append(sampling_result)

            # Convert positive bboxes to RoIs (adds batch indices) before ROI extractor
            # Use all sampled bboxes (pos + neg) to align with student's classification logits
            if sampling_results and any(res.bboxes.numel() > 0 for res in sampling_results):
                sampled_bboxes_list = [res.bboxes for res in sampling_results]
                rois = rbbox2roi(sampled_bboxes_list)  # shape (k, 6): [batch_ind, cx, cy, w, h, angle]
                teacher_roi_feats = self.teacher_model.roi_head.bbox_roi_extractor(
                    teacher_x[:self.teacher_model.roi_head.bbox_roi_extractor.num_inputs],
                    rois)
                teacher_cls_score, _ = self.teacher_model.roi_head.bbox_head(teacher_roi_feats)


        # Student forward
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                            self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
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

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 teacher_cls_score=teacher_cls_score,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation (delegate to parent)."""
        return super().simple_test(img, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test with augmentations (delegate to parent)."""
        return super().aug_test(imgs, img_metas, **kwargs)

    @property
    def with_rpn(self):
        return super().with_rpn

    @property
    def with_roi_head(self):
        return super().with_roi_head
