import torch
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader


class Evaluator:
    def __init__(self, model, dataset, device="cuda", batch_size=2):
        self.model = model
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self.device = device

    def evaluate_micro_f1(self):
        self.model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for batch in self.loader:
                batch = batch.to(self.device)
                out = self.model(batch.x, batch.edge_index)
                ys.append(batch.y.cpu())
                preds.append((torch.sigmoid(out) > 0.5).int().cpu())
        y_true = torch.cat(ys, dim=0).numpy()
        y_pred = torch.cat(preds, dim=0).numpy()
        micro_f1 = f1_score(y_true, y_pred, average="micro")
        print(f"Micro-F1: {micro_f1:.4f}")
