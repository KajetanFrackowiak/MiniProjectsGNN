import argparse
import json
import os
import secrets
from datetime import datetime
from typing import cast

import torch
import wandb
from evaluate import Evaluator
from gat import GAT
from load_data import load_data
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from training import Trainer
from utils import load_hyperparameters, plot_training_stats, save_metadata, set_seed


def save_stats(train_stats, eval_stats, filepath):
    combined_stats = {"training_stats": train_stats, "evaluation_stats": eval_stats}
    with open(filepath, "w") as f:
        json.dump(combined_stats, f)


def main():
    parser = argparse.ArgumentParser(description="GAT Node Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Datasets: PPI, Cora, CiteSeer, PubMed",
    )
    args = parser.parse_args()

    seed = secrets.randbelow(2**32)
    set_seed(seed)
    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Using seed: {seed}, datetime: {datetime_now}")

    # Define outputs directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(root_dir, "outputs")

    wandb.init(project="GAT_Node_Classification", name=f"GAT_{args.dataset}_{datetime_now}")
    wandb.config.update({"seed": seed, "datetime_now": datetime_now})

    config = load_hyperparameters("hyperparameters.yaml")
    wandb.config.update(config)

    train_dataset, val_dataset, test_dataset = load_data(args.dataset)

    if isinstance(train_dataset, Data):
        in_channels = train_dataset.num_node_features
        out_channels = len(torch.unique(train_dataset.y))
    else:
        first_data = cast(Data, train_dataset[0])
        in_channels = first_data.num_node_features
        out_channels = cast(torch.Tensor, first_data.y).shape[1]

    model = GAT(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        out_channels=out_channels,
        dropout=config["dropout"],
        num_heads=config["num_heads"],
    )

    model_metadata_dir = os.path.join(outputs_dir, "model_metadata")
    os.makedirs(model_metadata_dir, exist_ok=True)
    save_metadata(
        model,
        config,
        seed=seed,
        datetime_now=datetime_now,
        filepath=os.path.join(
            model_metadata_dir, f"model_metadata_{args.dataset}_{datetime_now}.json"
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Only move transductive Data objects to device; inductive Datasets handle it during iteration
    if (
        isinstance(train_dataset, Data)
        and isinstance(val_dataset, Data)
        and isinstance(test_dataset, Data)
    ):
        train_dataset = train_dataset.to(str(device))
        val_dataset = val_dataset.to(str(device))
        test_dataset = test_dataset.to(str(device))

    # Wrap inductive datasets in DataLoader for Trainer
    if isinstance(train_dataset, Data) and isinstance(val_dataset, Data):
        trainer_train_dataset = train_dataset
        trainer_val_dataset = val_dataset
    else:
        trainer_train_dataset = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        trainer_val_dataset = DataLoader(
            val_dataset, batch_size=config["batch_size"], shuffle=False
        )

    trainer = Trainer(
        model,
        trainer_train_dataset,
        trainer_val_dataset,
        dataset_name=args.dataset,
        datetime_now=datetime_now,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_stats = trainer.train(epochs=config["epochs"])

    # Wrap inductive datasets in DataLoader for Evaluator
    if isinstance(test_dataset, Data):
        evaluator_test_dataset = test_dataset
    else:
        evaluator_test_dataset = DataLoader(test_dataset, batch_size=config["batch_size"])

    evaluator = Evaluator(
        model, evaluator_test_dataset, device=str(device), batch_size=config["batch_size"]
    )
    if isinstance(test_dataset, Data):
        eval_stats = evaluator.evaluate_accuracy_transductive()
    else:
        eval_stats = evaluator.evaluate_micro_f1_inductive()

    train_eval_stats_dir = os.path.join(outputs_dir, "train_eval_stats")
    os.makedirs(train_eval_stats_dir, exist_ok=True)
    save_stats(
        train_stats,
        eval_stats,
        filepath=os.path.join(
            train_eval_stats_dir, f"train_eval_stats_{args.dataset}_{datetime_now}.json"
        ),
    )

    plot_training_stats(
        train_stats,
        dataset_name=args.dataset,
        datetime_now=datetime_now,
        output_dir=os.path.join(outputs_dir, "training_plots"),
    )


if __name__ == "__main__":
    main()
