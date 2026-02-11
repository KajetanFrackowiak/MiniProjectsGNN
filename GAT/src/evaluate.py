from typing import cast

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        dataset: Data | DataLoader,
        device: str = "cuda",
        batch_size: int = 2,
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        if isinstance(dataset, Data):
            self.is_transductive = True
        else:
            self.is_transductive = False
            # If dataset is already a DataLoader, use it directly
            if isinstance(dataset, DataLoader):
                self.loader = dataset
            else:
                self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def evaluate_accuracy_transductive(self) -> float:
        self.model.eval()
        # Hint to type checker that self.dataset is Data here
        data = cast(Data, self.dataset)

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            out = torch.nn.functional.log_softmax(out, dim=1)

            # Cast optional attributes to Tensor for type safety
            mask = cast(torch.Tensor, data.test_mask)
            y = cast(torch.Tensor, data.y)

            pred = out[mask].max(1)[1]
            correct = pred.eq(y[mask]).sum().item()
            acc = correct / mask.sum().item()
        print(f"Accuracy: {acc:.4f}")
        return acc

    def evaluate_micro_f1_inductive(self) -> float:
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            # Multiple graphs, multi-label (e.g., PPI) - no masks
            for batch in self.loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                pred = (torch.sigmoid(out) > 0.5).int().cpu().numpy()
                y = batch.y.cpu().numpy()
                preds.append(pred)
                ys.append(y)
        y_true = np.concatenate(ys)
        y_pred = np.concatenate(preds)
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        print(f"Micro-F1: {micro_f1:.4f}")
        return float(micro_f1)
