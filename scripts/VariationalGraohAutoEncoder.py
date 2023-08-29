from typing import List, Tuple

import torch
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VariationalGraohAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels_list[0])
        self.conv_list = [
            GCNConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]

        self.conv_mu = GCNConv(hidden_channels_list[-1], out_channels)
        self.conv_logstd = GCNConv(hidden_channels_list[-1], out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index).relu()
        for conv in self.conv_list:
            x = conv(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
