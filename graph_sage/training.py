import torch
from tqdm import tqdm
import wandb
import os
from torch_geometric.loader import DataLoader


class Trainer:
    def __init__(
        self,
        model,
        dataset_train,
        dataset_val,
        dataset_name,
        datetime_now,
        learning_rate=0.01,
        weight_decay=5e-4,
        batch_size=2,
    ):
        self.model = model
        self.dataset_name = dataset_name
        self.datetime_now = datetime_now

        # DataLoaders for training - PyTorch Geometric handles graph batching
        self.train_loader = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_stats = {"losses": [], "val_losses": [], "epochs": []}

    def train(self, epochs=200, device="cuda"):
        self.model.to(device)
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                batch = batch.to(device)
                self.optimizer.zero_grad()
                out = self.model(batch.x, batch.edge_index)
                loss = self.criterion(out, batch.y.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_loader)
            val_loss = self.evaluate(self.val_loader, device)
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                os.makedirs("model_checkpoints", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                    },
                    f"model_checkpoints/model_checkpoint_{self.dataset_name}_{self.datetime_now}_epoch_{epoch}.pt",
                )

            wandb.log(
                {"Train Loss": avg_train_loss, "Val Loss": val_loss, "Epoch": epoch}
            )
            self.train_stats["losses"].append(avg_train_loss)
            self.train_stats["val_losses"].append(val_loss)
            self.train_stats["epochs"].append(epoch)

        torch.save(
            self.model.state_dict(),
            f"model_checkpoints/model_final_{self.dataset_name}_{self.datetime_now}.pt",
        )
        
        return self.train_stats

    def evaluate(self, loader, device="cuda"):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = self.model(batch.x, batch.edge_index)
                loss = self.criterion(out, batch.y.float())
                total_loss += loss.item()
        return total_loss / len(loader)
