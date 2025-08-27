# Knowledge Distillation with LSKNet

This document provides a guide on how to use knowledge distillation (KD) within this repository, specifically for distilling knowledge from a larger teacher model (`lsk_s`) to a smaller student model (`lsk_t`).

## 1. Introduction to Knowledge Distillation

Knowledge Distillation is a model compression technique where a small "student" model is trained to mimic the behavior of a larger, pre-trained "teacher" model. The goal is to transfer the "knowledge" from the teacher to the student, enabling the student to achieve better performance than it would if trained from scratch on the same data.

In this implementation, we use Hinton's Knowledge Distillation method, which focuses on the classification task. The student model is trained on a composite loss function that includes:
1.  **Standard Loss**: The regular loss function computed against the ground truth labels (e.g., Cross-Entropy for classification and Smooth L1 for regression).
2.  **Distillation Loss**: A loss function that encourages the student's output logits to match the teacher's output logits. We use the Kullback-Leibler (KL) Divergence loss for this purpose.

## 2. Implementation Details

### Key Components

-   **`KDOrientedRCNN` (`mmrotate/models/detectors/kd_oriented_rcnn.py`)**: A new detector that manages both the teacher and student models. It ensures that the teacher's weights are frozen and only used for generating "soft labels" during training.
-   **`KnowledgeDistillationKLDivLoss` (`mmrotate/models/losses/kd_kl_div_loss.py`)**: The KL-divergence loss function used to measure the difference between the teacher's and student's probability distributions.
-   **Modified `RotatedShared2FCBBoxHead`**: The bounding box head has been updated to accept the teacher's logits and compute the distillation loss alongside the standard classification loss.

### How it Works

1.  During training, the `KDOrientedRCNN` detector performs a forward pass with both the teacher and student models.
2.  The teacher model, with its weights frozen, processes the input images to generate classification scores (logits) for the detected objects.
3.  The student model also processes the input images and generates its own logits.
4.  The student's logits are used to calculate two losses:
    -   The standard classification loss against the ground truth labels.
    -   The distillation loss (`loss_distill`) against the teacher's logits.
5.  The regression loss is calculated as usual, without any distillation.
6.  The total loss is a weighted sum of the classification, regression, and distillation losses. Only the student's weights are updated during backpropagation.

## 3. How to Run Distillation Training

### Configuration File

A new configuration file has been created to facilitate distillation training: `configs/lsknet/lsk_distill_s_t_fpn_1x_dota_le90.py`.

This file defines:
-   The **teacher model** (`lsk_s`) and the path to its pre-trained weights.
-   The **student model** (`lsk_t`).
-   The `KDOrientedRCNN` as the main model type.
-   The distillation loss (`loss_distill`) in the student's bounding box head configuration.

### Training Command

To start the distillation training, use the standard `train.py` script with the new configuration file:

```bash
# For single-GPU training
python tools/train.py configs/lsknet/lsk_distill_s_t_fpn_1x_dota_le90.py

# For multi-GPU training
./tools/dist_train.sh configs/lsknet/lsk_distill_s_t_fpn_1x_dota_le90.py <NUM_GPUS>
```

### Important Notes

-   **Teacher Checkpoint**: Make sure the `teacher_ckpt` path in the configuration file points to the correct pre-trained weights of the `lsk_s` model.
-   **Hyperparameters**: The distillation loss weight (`loss_weight`) and the temperature (`T`) in the `loss_distill` configuration can be tuned to achieve optimal performance.
-   **Regression Task**: This implementation focuses on distilling knowledge for the classification task. The regression task is learned directly from the ground truth data, as distilling regression knowledge is less straightforward.
"""
