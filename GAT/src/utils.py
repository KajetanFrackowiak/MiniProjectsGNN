import json
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml

matplotlib.use("Agg")  # Use a non-interactive backend for plotting


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_training_stats(train_stats, dataset_name, datetime_now, output_dir):
    """Plot training and validation loss over epochs.

    Args:
        train_stats (dict): Dictionary containing 'losses', 'val_losses', and 'epochs'.
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_stats["epochs"], train_stats["losses"], label="Training Loss")
    plt.plot(train_stats["epochs"], train_stats["val_losses"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        os.path.join(output_dir, f"training_validation_loss_{dataset_name}_{datetime_now}.png")
    )
    plt.close()


def load_hyperparameters(config_path: str) -> dict[str, Any]:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def save_stats(stats, filepath):
    with open(filepath, "w") as f:
        json.dump(stats, f)


def save_metadata(
    model: nn.Module,
    config: dict[str, Any],
    seed: int,
    datetime_now: str,
    filepath: str,
) -> None:
    # p.numel() gives the number of elements in the tensor
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params

    metadata = {
        "timestamp": datetime_now,
        "seed": seed,
        "model_parameters": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
        },
        "config": config,
    }

    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {filepath}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")
