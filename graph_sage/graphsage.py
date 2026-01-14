import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=121, dropout=0.5, aggr="mean"):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, SAGEConv):
                torch.nn.init.xavier_uniform_(m.lin_l.weight)
                torch.nn.init.xavier_uniform_(m.lin_r.weight)
                if m.lin_l.bias is not None:
                    torch.nn.init.zeros_(m.lin_l.bias)
                if m.lin_r.bias is not None:
                    torch.nn.init.zeros_(m.lin_r.bias)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # Multi-label task → raw logits for BCEWithLogitsLoss
