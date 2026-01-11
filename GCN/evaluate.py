class Evaluator:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def evaluate(self):
        self.model.eval()
        # self.model(self.data).shape = (num_nodes, num_classes) where num_classes is the number of classes that model predicts for each node
        _, pred = self.model(self.data).max(dim=1)
        correct = int(
            pred[self.data.test_mask].eq(self.data.y[self.data.test_mask]).sum().item()
        )
        acc = correct / int(self.data.test_mask.sum())
        print(f"Accuracy: {acc:.4f}")
        return acc
