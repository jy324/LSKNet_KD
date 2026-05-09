import torch
import torch.nn.functional as F
from mmrotate.core import rbbox2roi
from ..oriented_standard_roi_head import OrientedStandardRoIHead
from ...builder import ROTATED_HEADS


@ROTATED_HEADS.register_module()
class DistillOrientedStandardRoIHead(OrientedStandardRoIHead):
    """OrientedStandardRoIHead with pluggable knowledge distillation.

    Args:
        kd_cfg (dict|None): Configuration for KD. Example:
            kd_cfg=dict(
                enable=True,
                type='kl',         # currently only 'kl'
                T=4.0,             # temperature
                weight=0.5,        # loss weight
                align='truncate',  # 'truncate' | 'pad' (future) | 'none'
                positive_only=False,  # only distill positive samples
                detach_teacher=True,
            )
    Notes:
        - We DO NOT modify parent class logic; we call super().forward_train
          WITHOUT teacher logits so that base head produces normal losses.
        - Then we re-run (lightweight) a KD head calc using cached logits.
        - Because parent class currently in repo has inline KD, you should
          remove or guard that section to avoid double counting if both used.
    """

    def __init__(self, kd_cfg=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_cfg = kd_cfg or dict(enable=False)
        # internal cache for last forward student cls logits (before loss)
        self._last_student_cls = None
        self._last_pos_mask = None  # bool mask over samples used for optional positive_only KD

    # override only bbox forward train to capture student cls logits cleanly
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, teacher_cls_score=None):
        # call parent to get bbox_results (will also compute loss via bbox_head.loss)
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)
        # cache raw student cls for KD (before softmax)
        self._last_student_cls = bbox_results['cls_score'] if 'cls_score' in bbox_results else None
        # build positive mask (concatenate per-image pos/neg ordering produced by sampler)
        # sampling_results[i].bboxes = [pos_bboxes; neg_bboxes] in mmdet convention
        if sampling_results:
            pos_lens = [res.pos_bboxes.size(0) for res in sampling_results]
            neg_lens = [res.neg_bboxes.size(0) for res in sampling_results]
            masks = []
            for p, n in zip(pos_lens, neg_lens):
                if p + n == 0:
                    masks.append(torch.zeros(0, dtype=torch.bool, device=bbox_results['cls_score'].device))
                else:
                    m = torch.zeros(p + n, dtype=torch.bool, device=bbox_results['cls_score'].device)
                    m[:p] = True
                    masks.append(m)
            self._last_pos_mask = torch.cat(masks, dim=0) if masks else None
        else:
            self._last_pos_mask = None
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets,
                                        teacher_cls_score=None)  # parent KD disabled here
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      teacher_cls_score=None,
                      **kwargs):
        # Run parent logic but disable its inline KD by not passing teacher logits further.
        losses = super().forward_train(x, img_metas, proposal_list, gt_bboxes, gt_labels,
                                       gt_bboxes_ignore=gt_bboxes_ignore, gt_masks=gt_masks,
                                       )

        # Apply KD AFTER base losses.
        if self.kd_cfg.get('enable', False) and teacher_cls_score is not None and self._last_student_cls is not None:
            kd_losses = self._compute_kd(self._last_student_cls, teacher_cls_score, self.kd_cfg)
            losses.update(kd_losses)
        return losses

    def _compute_kd(self, student_logits, teacher_logits, cfg):
        losses = {}
        # shape align
        if student_logits.size(1) != teacher_logits.size(1):
            # class dimension mismatch -> skip
            losses['loss_kd_cls'] = student_logits.sum() * 0
            return losses
        # sample dimension align
        if cfg.get('align', 'truncate') == 'truncate':
            n = min(student_logits.size(0), teacher_logits.size(0))
            student_logits = student_logits[:n]
            teacher_logits = teacher_logits[:n]
        # optional: positive_only (need labels mask -> currently not stored; placeholder for future)
        if cfg.get('positive_only', False) and self._last_pos_mask is not None:
            # ensure mask length compatibility after truncate
            mask = self._last_pos_mask[:student_logits.size(0)]
            if mask.any():
                student_logits = student_logits[mask]
                teacher_logits = teacher_logits[mask]
            else:
                losses['loss_kd_cls'] = student_logits.sum() * 0
                return losses
        # temperature
        T = float(cfg.get('T', 1.0))
        detach_teacher = cfg.get('detach_teacher', True)
        if detach_teacher:
            teacher_logits = teacher_logits.detach()
        log_p = F.log_softmax(student_logits / T, dim=1)
        q = F.softmax(teacher_logits / T, dim=1)
        kd = F.kl_div(log_p, q, reduction='batchmean') * (T ** 2)
        weight = float(cfg.get('weight', 1.0))
        losses['loss_kd_cls'] = kd * weight
        return losses

