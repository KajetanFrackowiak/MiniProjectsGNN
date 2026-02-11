import os
from typing import cast

import torch
import torch.nn as nn
import wandb
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class to handle the training process of the GAT model."""

    def __init__(
        self,
        model: nn.Module,
        train_data: Data | DataLoader,
        val_data: Data | DataLoader,
        dataset_name: str,
        datetime_now: str,
        device: torch.device,
        learning_rate: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 100,
    ):
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.dataset_name = dataset_name
        self.datetime_now = datetime_now
        self.patience = patience
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        # Use different loss functions for transductive vs inductive
        if isinstance(self.train_data, Data):
            self.criterion = torch.nn.NLLLoss()
            self.is_transductive = True
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.is_transductive = False

        self.train_stats = {"losses": [], "val_losses": [], "epochs": []}

        # Move data to device
        if isinstance(self.train_data, Data):
            self.train_data = self.train_data.to(str(self.device))
            self.val_data = cast(Data, self.val_data).to(str(self.device))

    def train(self, epochs: int = 200) -> dict[str, list[float]]:
        self.model.train()
        best_val_loss = float("inf")
        counter = 0
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.optimizer.zero_grad()
            if isinstance(self.train_data, Data):
                # Transductive setting (e.g., Planetoid datasets)
                out = self.model(self.train_data.x, self.train_data.edge_index)
                out = torch.nn.functional.log_softmax(out, dim=1)
                loss = self.criterion(
                    out[self.train_data.train_mask],
                    cast(torch.Tensor, self.train_data.y)[self.train_data.train_mask],
                )
            else:
                # Inductive setting (e.g., PPI) - no masks, use entire graph
                loss = torch.tensor(0.0, device=self.device)
                # Iterate over each graph in the training set
                for graph in self.train_data:
                    graph = graph.to(self.device)
                    out = self.model(graph.x, graph.edge_index)
                    # For multi-label, output is already logits (no log_softmax applied)
                    loss = loss + self.criterion(out, graph.y.float())
                loss = loss / len(self.train_data)
            loss.backward()
            self.optimizer.step()

            # Compute validation loss
            self.model.eval()
            with torch.no_grad():
                if isinstance(self.val_data, Data):
                    val_out = self.model(self.val_data.x, self.val_data.edge_index)
                    val_out = torch.nn.functional.log_softmax(val_out, dim=1)
                    val_loss = self.criterion(
                        val_out[self.val_data.val_mask],
                        cast(torch.Tensor, self.val_data.y)[self.val_data.val_mask],
                    )
                else:
                    val_loss = torch.tensor(0.0, device=self.device)
                    # Iterate over each graph in the validation set
                    for graph in self.val_data:
                        graph = graph.to(self.device)
                        val_out = self.model(graph.x, graph.edge_index)
                        # For multi-label, output is already logits (no log_softmax applied)
                        val_loss = val_loss + self.criterion(val_out, graph.y.float())
                    val_loss = val_loss / len(self.val_data)
            self.model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss.item(),
                    },
                    f"checkpoints/best_model_{self.dataset_name}_{self.datetime_now}.pt",
                )
            else:
                counter += 1

            if counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )

            wandb.log(
                {
                    "Train Loss": loss.item(),
                    "Validation Loss": val_loss.item(),
                    "Epoch": epoch,
                }
            )

            self.train_stats["losses"].append(loss.item())
            self.train_stats["val_losses"].append(val_loss.item())
            self.train_stats["epochs"].append(epoch)

        return self.train_stats
