import argparse
import os
import secrets
import json
import wandb
import torch
from torch_geometric.data import Data

from load_data import load_data
from gat import GAT
from training import Trainer
from evaluate import Evaluator
from utils import load_hyperparameters, plot_training_stats, save_metadata, set_seed
from datetime import datetime


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

    wandb.init(
        project="GAT_Node_Classification", name=f"GAT_{args.dataset}_{datetime_now}"
    )
    wandb.config.update({"seed": seed, "datetime_now": datetime_now})

    config = load_hyperparameters("hyperparameters.yaml")
    wandb.config.update(config)

    train_dataset, val_dataset, test_dataset = load_data(args.dataset)

    if isinstance(train_dataset, Data):
        in_channels = train_dataset.num_node_features
        out_channels = len(torch.unique(train_dataset.y))
    else:
        in_channels = train_dataset[0].num_node_features
        out_channels = train_dataset[0].y.shape[1]

    model = GAT(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        out_channels=out_channels,
        dropout=config["dropout"],
        num_heads=config["num_heads"],
    )

    os.makedirs("model_metadata", exist_ok=True)
    save_metadata(
        model,
        config,
        seed=seed,
        datetime_now=datetime_now,
        filepath=f"model_metadata/model_metadata_{args.dataset}_{datetime_now}.json",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Only move transductive Data objects to device; inductive Datasets handle it during iteration
    if isinstance(train_dataset, Data):
        train_dataset = train_dataset.to(device)
        val_dataset = val_dataset.to(device)
        test_dataset = test_dataset.to(device)

    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        dataset_name=args.dataset,
        datetime_now=datetime_now,
        device=device,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    train_stats = trainer.train(epochs=config["epochs"])

    evaluator = Evaluator(
        model, test_dataset, device=device, batch_size=config["batch_size"]
    )
    if isinstance(test_dataset, Data):
        eval_stats = evaluator.evaluate_accuracy_transductive()
    else:
        eval_stats = evaluator.evaluate_micro_f1_inductive()

    os.makedirs("train_eval_stats", exist_ok=True)
    save_stats(
        train_stats,
        eval_stats,
        filepath=f"train_eval_stats/train_eval_stats_{args.dataset}_{datetime_now}.json",
    )

    plot_training_stats(
        train_stats,
        dataset_name=args.dataset,
        datetime_now=datetime_now,
        output_dir="training_plots",
    )


if __name__ == "__main__":
    main()
