import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, out_channels=7, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

        # Glorot (Xavier) initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Glorot (Xavier) uniform initialization"""
        for m in self.modules():
            if isinstance(m, GCNConv):
                # "lin" is the internal linear layer of GCNConv
                if hasattr(m, "lin"):
                    torch.nn.init.xavier_uniform_(m.lin.weight)
                    if m.lin.bias is not None:
                        torch.nn.init.zeros_(m.lin.bias)

    def forward(self, data):
        # data.shape = (num_nodes, num_node_features)
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Log softmax + NLLLoss in training = CrossEntropyLoss
        return F.log_softmax(x, dim=1)
