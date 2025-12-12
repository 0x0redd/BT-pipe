import json
import os
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _build_transforms(task: str, is_train: bool) -> T.Compose:
    base_transforms: List[Any] = [T.ToTensor()]

    if is_train:
        # Light augmentations; detection models expect tensors in [0,1]
        base_transforms.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    if task == "classification":
        # Normalize for ImageNet pretrained backbones
        base_transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    return T.Compose(base_transforms)


class TumorDataset(Dataset):
    """
    Dataset for both detection and classification.

    Expects a JSON file containing a list of samples:
    {
      "image": "path/to/img.png",
      "label": "Glioma",
      "bbox": [x1, y1, x2, y2]   # optional for classification
    }
    """

    def __init__(
        self,
        annotation_path: str,
        image_root: str,
        labels: List[str],
        task: str = "detection",
        transforms: T.Compose | None = None,
    ) -> None:
        super().__init__()
        self.annotation_path = annotation_path
        self.image_root = image_root
        self.labels = labels
        self.label_to_idx = {name: i for i, name in enumerate(labels)}
        self.task = task
        self.transforms = transforms or _build_transforms(task, is_train=True)

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_samples: List[Dict[str, Any]] = json.load(f)

        if not isinstance(raw_samples, list):
            raise ValueError(f"Annotation file must contain a list: {annotation_path}")

        # Filter out samples with invalid bounding boxes (zero width or height)
        self.samples: List[Dict[str, Any]] = []
        invalid_count = 0
        for sample in raw_samples:
            if task == "detection" and "bbox" in sample:
                bbox = sample["bbox"]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        self.samples.append(sample)
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            else:
                self.samples.append(sample)
        
        if invalid_count > 0:
            print(f"Warning: Filtered out {invalid_count} samples with invalid bounding boxes (zero width or height)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        record = self.samples[idx]
        image_path = os.path.join(self.image_root, record["image"])
        image = Image.open(image_path).convert("RGB")

        target_label = record.get("label")
        target_bbox = record.get("bbox")

        if self.transforms:
            image = self.transforms(image)

        if self.task == "classification":
            if target_label is None:
                raise ValueError("Classification task requires 'label' in annotation.")
            label_idx = self.label_to_idx[target_label]
            return image, torch.tensor(label_idx, dtype=torch.long)

        if target_bbox is None:
            raise ValueError("Detection task requires 'bbox' in annotation.")

        boxes = torch.tensor([target_bbox], dtype=torch.float32)
        labels = torch.tensor(
            [self.label_to_idx[target_label]], dtype=torch.int64
        )  # detection labels start at 0

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image, target


def detection_collate_fn(batch: List[Tuple[Any, Any]]) -> Tuple[List[Any], List[Any]]:
    images, targets = zip(*batch)
    return list(images), list(targets)

