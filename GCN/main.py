import argparse
import os
import secrets

import wandb
from load_data import load_data
from gcn import GCN
from training import Trainer
from evaluate import Evaluator
from utils import load_hyperparameters, plot_training_stats, save_metadata, set_seed
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="GCN Node Classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Name of the dataset (Cora, Citeseer, Pubmed)",
    )
   
    seed = secrets.randbelow(2**32)
    set_seed(seed)
    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    args = parser.parse_args()
    wandb.init(project="GCN_Node_Classification", name=f"GCN_{args.dataset}_{datetime_now}")
    wandb.config.update({"seed": seed})

    config = load_hyperparameters("hyperparameters.yaml")
    wandb.config.update(config)

    data = load_data(args.dataset)

    out_channels = data.y.max().item() + 1
    model = GCN(
        in_channels=data.num_node_features,
        hidden_channels=config["hidden_channels"],
        out_channels=out_channels,
        dropout=config["dropout"],
    )
    os.makedirs("model_metadata", exist_ok=True)
    save_metadata(
        model,
        config,
        seed=seed,
        datetime_now=datetime_now,
        filepath=f"model_metadata/model_metadata_{args.dataset}_{datetime_now}.json",
    )

    trainer = Trainer(
        model,
        data,
        dataset_name=args.dataset,
        datetime_now=datetime_now,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        )
    train_stats = trainer.train(epochs=config["epochs"])

    evaluator = Evaluator(model, data)
    evaluator.evaluate()

    plot_training_stats(train_stats, dataset_name=args.dataset, datetime_now=datetime_now, output_dir="training_plots")


if __name__ == "__main__":
    main()
