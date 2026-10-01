import argparse
from pathlib import Path

import torch

from homework.metrics import DetectionMetric
from homework.models import Detector, save_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if (data_dir / "train").is_dir() and (data_dir / "val").is_dir():
        return data_dir
    nested_dir = data_dir / data_dir.name
    if (nested_dir / "train").is_dir() and (nested_dir / "val").is_dir():
        return nested_dir
    raise FileNotFoundError(f"Could not find detection data under {data_dir}")


def run_epoch(model, data, device, optimizer=None, depth_weight=1.0):
    training = optimizer is not None
    model.train(training)
    metric = DetectionMetric()
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for batch in data:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, depth = model(batch["image"])
            segmentation_loss = torch.nn.functional.cross_entropy(logits, batch["track"])
            depth_loss = torch.nn.functional.smooth_l1_loss(depth, batch["depth"])
            loss = segmentation_loss + depth_weight * depth_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = batch["image"].size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            metric.add(logits.argmax(dim=1), batch["track"], depth, batch["depth"])

    values = metric.compute()
    values["loss"] = total_loss / total_samples
    return values


def train(
    data_dir="drive_data",
    epochs=30,
    batch_size=32,
    learning_rate=3e-4,
    num_workers=2,
    depth_weight=1.0,
    seed=2024,
):
    from homework.datasets.road_dataset import load_data

    torch.manual_seed(seed)
    device = get_device()
    data_dir = resolve_data_dir(data_dir)
    model = Detector().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_data = load_data(
        Path(data_dir) / "train",
        transform_pipeline="default",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_data = load_data(
        Path(data_dir) / "val",
        transform_pipeline="default",
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    for epoch in range(epochs):
        train_values = run_epoch(model, train_data, device, optimizer, depth_weight)
        val_values = run_epoch(model, val_data, device, depth_weight=depth_weight)
        scheduler.step()
        print(
            f"Epoch {epoch + 1:02d}/{epochs}: "
            f"train_loss={train_values['loss']:.4f} val_loss={val_values['loss']:.4f} "
            f"val_iou={val_values['iou']:.4f} val_depth={val_values['abs_depth_error']:.4f}"
        )

    output_path = save_model(model)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="drive_data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
