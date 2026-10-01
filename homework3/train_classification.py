import argparse
from pathlib import Path

import torch

from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric
from homework.models import Classifier, save_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_data_dir(data_dir):
    data_dir = Path(data_dir)
    if (data_dir / "train" / "labels.csv").exists():
        return data_dir
    nested_dir = data_dir / data_dir.name
    if (nested_dir / "train" / "labels.csv").exists():
        return nested_dir
    raise FileNotFoundError(f"Could not find classification data under {data_dir}")


def run_epoch(model, data, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    metric = AccuracyMetric()
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for image, label in data:
            image, label = image.to(device), label.to(device)
            logits = model(image)
            loss = torch.nn.functional.cross_entropy(logits, label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * label.size(0)
            metric.add(logits.argmax(dim=1), label)

    return total_loss / metric.total, metric.compute()["accuracy"]


def train(
    data_dir="classification_data",
    epochs=20,
    batch_size=128,
    learning_rate=3e-4,
    num_workers=2,
    seed=2024,
):
    torch.manual_seed(seed)
    device = get_device()
    data_dir = resolve_data_dir(data_dir)
    model = Classifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    train_data = load_data(
        Path(data_dir) / "train",
        transform_pipeline="aug",
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
        train_loss, train_accuracy = run_epoch(model, train_data, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_data, device)
        scheduler.step()
        print(
            f"Epoch {epoch + 1:02d}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    output_path = save_model(model)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="classification_data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    train(**vars(parser.parse_args()))
