"""Detector module package for tumor detection/classification."""

from .dataset import TumorDataset, detection_collate_fn
from .model_factory import build_model

__all__ = ["TumorDataset", "detection_collate_fn", "build_model"]

