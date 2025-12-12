import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import yaml

from models.detector import TumorDataset, detection_collate_fn, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tumor detector/classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(cfg: Dict, task: str) -> Tuple[DataLoader, DataLoader]:
    from models.detector.dataset import _build_transforms  # lazy import to avoid circularity

    train_tfms = _build_transforms(task, is_train=True)
    val_tfms = _build_transforms(task, is_train=False)

    train_dataset = TumorDataset(
        annotation_path=cfg["data"]["train_annotations"],
        image_root=cfg["data"]["image_root"],
        labels=cfg["data"]["labels"],
        task=task,
        transforms=train_tfms,
    )
    val_dataset = TumorDataset(
        annotation_path=cfg["data"]["val_annotations"],
        image_root=cfg["data"]["image_root"],
        labels=cfg["data"]["labels"],
        task=task,
        transforms=val_tfms,
    )

    collate_fn = detection_collate_fn if task == "detection" else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    ckpt_dir: str,
    experiment_name: str,
) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = Path(ckpt_dir) / f"{experiment_name}_epoch{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


def train_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if cfg["train"].get("grad_clip"):
                clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate_classification(model, val_loader, device, criterion)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, cfg["train"]["checkpoint_dir"], cfg["experiment_name"]
            )


@torch.no_grad()
def evaluate_classification(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train_detection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict,
) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            if cfg["train"].get("grad_clip"):
                clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            running_loss += losses.item()

        avg_train_loss = running_loss / len(train_loader)

        val_loss = evaluate_detection(model, val_loader, device)
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f}")

        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, epoch, cfg["train"]["checkpoint_dir"], cfg["experiment_name"]
            )


@torch.no_grad()
def evaluate_detection(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / len(loader) if len(loader) > 0 else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = cfg["task"]
    num_classes = len(cfg["data"]["labels"])
    model = build_model(
        task=task,
        num_classes=num_classes,
        model_name=cfg["model"]["name"],
        pretrained=cfg["model"]["pretrained"],
    )
    model.to(device)

    train_loader, val_loader = make_dataloaders(cfg, task)

    print(
        f"Starting training | task={task} model={cfg['model']['name']} "
        f"num_classes={num_classes} device={device}"
    )

    if task == "classification":
        train_classification(model, train_loader, val_loader, device, cfg)
    elif task == "detection":
        train_detection(model, train_loader, val_loader, device, cfg)
    else:
        raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()

