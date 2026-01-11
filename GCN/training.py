import torch
import wandb
from tqdm import tqdm
import os

class Trainer:
    """
    Trainer class to handle training of the GCN model with early stopping and logging.
    """

    def __init__(
        self, model, data, dataset_name, datetime_now, learning_rate=0.01, weight_decay=5e-4
    ):
        self.model = model
        self.data = data
        self.dataset_name = dataset_name
        self.datetime_now = datetime_now
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = torch.nn.NLLLoss()

        self.train_stats = {"losses": [], "val_losses": [], "epochs": []}

    def train(self, epochs=200):
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = self.criterion(
                out[self.data.train_mask], self.data.y[self.data.train_mask]
            )
            loss.backward()
            self.optimizer.step()

            # Compute validation loss for early stopping
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(self.data)
                val_loss = self.criterion(
                    val_out[self.data.val_mask], self.data.y[self.data.val_mask]
                )
            self.model.train()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}"
                )
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"checkpoints/model_epoch_{epoch}_{self.dataset_name}_{self.datetime_now}.pt",
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
