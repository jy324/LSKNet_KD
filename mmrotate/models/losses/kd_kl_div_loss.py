import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models.losses import weighted_loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred, soft_label, T, detach_target=True):
    """KLDiv loss with temperature for distillation (Hinton KD).

    Args:
        pred (Tensor): Student logits (N, C).
        soft_label (Tensor): Teacher logits (N, C).
        T (float): Temperature.
        detach_target (bool): Detach teacher logits. Default True.
    """
    assert pred.size() == soft_label.size()
    if detach_target:
        soft_label = soft_label.detach()
    log_p = F.log_softmax(pred / T, dim=1)
    q = F.softmax(soft_label / T, dim=1)
    # scale by T^2 per Hinton et al.
    return F.kl_div(log_p, q, reduction='batchmean') * (T ** 2)


@ROTATED_LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """KLDiv distillation loss.

    Args:
        T (float): Temperature.
        loss_weight (float): Weight factor.
        detach_target (bool): Whether to detach teacher logits.
    """

    def __init__(self, T=2.0, loss_weight=1.0, detach_target=True):
        super().__init__()
        self.T = T
        self.loss_weight = loss_weight
        self.detach_target = detach_target

    def forward(self, pred, soft_label, **kwargs):  # kwargs kept for compat
        loss = knowledge_distillation_kl_div_loss(
            pred, soft_label, self.T, detach_target=self.detach_target)
        return self.loss_weight * loss
