import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 8,
        num_heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_channels * num_heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers in GATConv with Xavier uniform
        for m in self.modules():
            if isinstance(m, GATConv):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        torch.nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        torch.nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # Return raw logits for multi-label (e.g., PPI) or log_softmax for single-label
        # The calling code should determine which to use based on the task
        return x
