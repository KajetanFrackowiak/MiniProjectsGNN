import argparse
import os
import secrets
import wandb

from load_data import load_data
from graphsage import GraphSAGE
from training import Trainer
from evaluate import Evaluator
from utils import load_hyperparameters, plot_training_stats, save_metadata, set_seed
from datetime import datetime


def main():
    seed = secrets.randbelow(2**32)
    set_seed(seed)
    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")

    wandb.init(project="GCN_Node_Classification", name=f"GCN_PPI_{datetime_now}")
    wandb.config.update({"seed": seed})

    config = load_hyperparameters("hyperparameters.yaml")
    wandb.config.update(config)

    train_dataset, val_dataset, test_dataset = load_data()

    # For multi-label task, out_channels should match the label dimension
    out_channels = train_dataset[0].y.shape[1]

    model = GraphSAGE(
        in_channels=int(train_dataset[0].num_node_features),
        hidden_channels=config["hidden_channels"],
        out_channels=out_channels,
        dropout=config["dropout"],
        aggr=config["sage_aggr"],
    )

    os.makedirs("model_metadata", exist_ok=True)
    save_metadata(
        model,
        config,
        seed=seed,
        datetime_now=datetime_now,
        filepath=f"model_metadata/model_metadata_PPI_{datetime_now}.json",
    )

    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        dataset_name="PPI",
        datetime_now=datetime_now,
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
    )
    train_stats = trainer.train(epochs=config["epochs"])

    evaluator = Evaluator(model, test_dataset, batch_size=config["batch_size"])
    evaluator.evaluate_micro_f1()

    plot_training_stats(
        train_stats,
        dataset_name="PPI",
        datetime_now=datetime_now,
        output_dir="training_plots",
    )


if __name__ == "__main__":
    main()
