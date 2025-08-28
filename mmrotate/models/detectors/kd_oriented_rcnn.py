import torch
from mmcv.runner import load_checkpoint

from ..builder import ROTATED_DETECTORS, build_detector
from .two_stage import RotatedTwoStageDetector


@ROTATED_DETECTORS.register_module()
class KDOrientedRCNN(RotatedTwoStageDetector):
    """Oriented R-CNN with classification logit distillation.

    Only adds a distillation loss on RoI cls logits; regression unchanged.
    """

    def __init__(self, teacher_config, teacher_ckpt, student, *args, **kwargs):
        super().__init__(
            backbone=student.backbone,
            neck=student.get('neck'),
            rpn_head=student.get('rpn_head'),
            roi_head=student.get('roi_head'),
            train_cfg=student.get('train_cfg'),
            test_cfg=student.get('test_cfg'),
            init_cfg=student.get('init_cfg'))

        self.teacher = build_detector(teacher_config)
        load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        # Forward teacher (no grad) to get cls logits on same proposals used by student
        with torch.no_grad():
            t_feats = self.teacher.extract_feat(img)
            if self.teacher.with_rpn:
                t_proposal_list = self.teacher.rpn_head.simple_test_rpn(t_feats, img_metas)
            else:
                t_proposal_list = proposals
            # Use teacher proposals for both teacher logits + student training for alignment
            t_rois = [p[:, :5] if p.size(-1) > 5 else p for p in t_proposal_list]
            # Build RoIs tensor
            from mmrotate.core import rbbox2roi
            t_rois_tensor = rbbox2roi(t_rois)
            t_bbox_results = self.teacher.roi_head._bbox_forward(t_feats, t_rois_tensor)
            teacher_cls_score = t_bbox_results['cls_score']

        # Student forward using same proposals for fair distill
        s_feats = self.extract_feat(img)
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                s_feats, img_metas, gt_bboxes, gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg)
        else:
            proposal_list = proposals
            rpn_losses = {}

        # Use proposal_list for sampling; pass teacher logits into bbox head loss
        roi_losses = self.roi_head.forward_train(
            s_feats, img_metas, proposal_list,
            gt_bboxes, gt_labels,
            gt_bboxes_ignore, gt_masks,
            teacher_cls_score=teacher_cls_score)

        losses = {}
        losses.update(rpn_losses)
        losses.update(roi_losses)
        return losses
