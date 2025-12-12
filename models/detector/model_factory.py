import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _build_classification(num_classes: int, pretrained: bool) -> nn.Module:
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _build_faster_rcnn(num_classes: int, pretrained: bool):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    # Replace the box predictor to match dataset classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_model(task: str, num_classes: int, model_name: str, pretrained: bool = True):
    if task == "classification":
        return _build_classification(num_classes=num_classes, pretrained=pretrained)
    if task == "detection":
        if "fasterrcnn" in model_name.lower():
            return _build_faster_rcnn(num_classes=num_classes, pretrained=pretrained)
        raise ValueError(f"Unsupported detection model: {model_name}")

    raise ValueError(f"Unsupported task: {task}")

