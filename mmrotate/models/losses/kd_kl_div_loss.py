import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


@weighted_loss
def knowledge_distillation_kl_div_loss(pred,
                                       soft_label,
                                       T,
                                       detach_target=True):
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, C).
        soft_label (Tensor): Target logits with shape (N, C).
        T (int): Temperature for distillation.
        detach_target (bool): Whether to detach the soft_label tensor.
            Defaults to True.
    Returns:
        Tensor: The calculated loss.
    """
    assert pred.size() == soft_label.size()
    if detach_target:
        soft_label = soft_label.detach()

    log_p = F.log_softmax(pred / T, dim=1)
    q = F.softmax(soft_label / T, dim=1)
    return F.kl_div(log_p, q, reduction='sum') * (T**2) / pred.size(0)


@LOSSES.register_module()
class KnowledgeDistillationKLDivLoss(nn.Module):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        T (int): Temperature for distillation.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        detach_target (bool): Whether to detach the soft_label tensor.
            Defaults to True.
    """

    def __init__(self,
                 T,
                 loss_weight=1.0,
                 reduction='mean',
                 detach_target=True):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        self.T = T
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.detach_target = detach_target

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits.
            soft_label (Tensor): Target logits.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_kd = self.loss_weight * knowledge_distillation_kl_div_loss(
            pred,
            soft_label,
            self.T,
            detach_target=self.detach_target)
        return loss_kd
"""
