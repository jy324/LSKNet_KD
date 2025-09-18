import random
from typing import List, Set

import numpy as np

from .dota import DOTADataset
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class BalancedDOTADataset(DOTADataset):
    """DOTA dataset with class-balanced image-level downsampling.

    After the original :class:`DOTADataset` loads all annotations into
    ``self.data_infos``, we optionally keep only a subset of images so that
    (as much as possible) each class contributes a similar number of images.

    Strategy (balance_mode='class'):
        1. Build mapping class -> list[image_index] (image contains >=1 instance).
        2. Sample ``target_per_class = target_images // num_classes`` images
           for every class (rare classes first so they are all kept).
        3. If quota not yet filled, randomly sample from the remaining pool.

    Args:
        ann_file (str): Path to annotation directory (same as DOTADataset).
        pipeline (list[dict]): Data pipeline.
        version (str): Angle representation, passed to parent.
        difficulty (int): Difficulty threshold, passed to parent.
        subset_ratio (float): Portion of images to keep (0 < r <= 1). If 1 or >=1
            no downsampling performed. Default: 1.0.
        balance_mode (str): 'class' for class-balanced sampling; 'random' for
            plain random sampling. Default: 'class'.
        seed (int): Random seed for reproducibility. Default: 0.
        **kwargs: Other keyword args passed to parent dataset.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 subset_ratio: float = 1.0,
                 balance_mode: str = 'class',
                 seed: int = 0,
                 **kwargs):
        self.subset_ratio = subset_ratio
        self.balance_mode = balance_mode
        self.seed = seed
        super().__init__(ann_file, pipeline, version=version, difficulty=difficulty, **kwargs)
        if 0 < self.subset_ratio < 1.0:
            self._apply_subset()

    def _apply_subset(self):
        total = len(self.data_infos)
        if total == 0:
            return
        target = max(1, int(round(total * self.subset_ratio)))
        if target >= total:  # nothing to do
            return

        random.seed(self.seed)
        np.random.seed(self.seed)

        if self.balance_mode == 'random':
            selected_idx = sorted(random.sample(range(total), target))
        elif self.balance_mode == 'class':  # class-balanced image sampling
            # Collect per-image class sets
            img_classes: List[Set[int]] = []
            for info in self.data_infos:
                labels_arr = info['ann']['labels']
                if isinstance(labels_arr, list):  # test split or empty
                    labels_set = set(labels_arr)
                else:
                    labels_set = set(labels_arr.tolist()) if labels_arr.size > 0 else set()
                img_classes.append(labels_set)

            num_classes = len(self.CLASSES)
            class_to_imgs = {c: [] for c in range(num_classes)}
            for idx, cls_set in enumerate(img_classes):
                for c in cls_set:
                    if c < num_classes:
                        class_to_imgs[c].append(idx)

            target_per_class = max(1, target // num_classes)
            selected = set()

            # Sample rare classes first (ascending by available images)
            for c, idxs in sorted(class_to_imgs.items(), key=lambda x: len(x[1])):
                if not idxs:
                    continue
                available = [i for i in idxs if i not in selected]
                if not available:
                    continue
                k = min(target_per_class, len(available))
                chosen = random.sample(available, k)
                selected.update(chosen)

            # Fill remaining quota from leftover images
            if len(selected) < target:
                remaining_pool = [i for i in range(total) if i not in selected]
                need = target - len(selected)
                if remaining_pool:
                    need = min(need, len(remaining_pool))
                    selected.update(random.sample(remaining_pool, need))

            # If (rare) overshoot, trim
            if len(selected) > target:
                selected = set(random.sample(list(selected), target))

            selected_idx = sorted(selected)

        # Replace data_infos & img_ids
        self.data_infos = [self.data_infos[i] for i in selected_idx]
        self.img_ids = [info['filename'][:-4] for info in self.data_infos]
        # Reset grouping flags if parent uses them
        if hasattr(self, '_set_group_flag'):
            self._set_group_flag()
        kept = len(self.data_infos)
        print(f'[BalancedDOTADataset] Kept {kept}/{total} = {kept/total:.2%} images (mode={self.balance_mode}, ratio={self.subset_ratio}).')
