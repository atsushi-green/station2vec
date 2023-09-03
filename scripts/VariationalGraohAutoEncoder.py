from typing import List, Tuple

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from StationData import CROSS_ENTROPY_INDEXES


# ショートカットコネクションを入れたほうがいいかもしれないが、オートエンコーダーの性質上、
# ショートカットコネクションを入れると予測が簡単になりすぎてしまうのではないかという懸念がある。
class VariationalGraohAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], out_channels: int):
        super().__init__()
        # self.conv1 = GCNConv(in_channels, hidden_channels_list[0])
        self.conv1 = SAGEConv(in_channels, hidden_channels_list[0])

        # self.conv_list = [
        #     GCNConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        # ]
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # 最終層だけGCNConvで隣接ノードを全て使う
        self.conv_mu = GCNConv(hidden_channels_list[-1], out_channels)
        self.conv_logstd = GCNConv(hidden_channels_list[-1], out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x, edge_index).relu()
        for conv in self.conv_list:
            x = conv(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class VariationalGraohAutoDecoder(torch.nn.Module):
    def __init__(self, embedding_channels: int, hidden_channels_list: List[int], out_channels: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # self.conv1 = GCNConv(embedding_channels, hidden_channels_list[0])
        # self.conv_list = [
        #     GCNConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        # ]
        self.conv1 = SAGEConv(embedding_channels, hidden_channels_list[0])
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # 最終層だけGCNConvで隣接ノードを全て使う
        self.conv_final = GCNConv(hidden_channels_list[-1], out_channels)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(z, edge_index).relu()
        for conv in self.conv_list:
            x = conv(x, edge_index).relu()
        x = self.conv_final(x, edge_index)
        # 0-1をとる変数にはシグモイド関数をかませる
        x[:, CROSS_ENTROPY_INDEXES] = self.sigmoid(x[:, CROSS_ENTROPY_INDEXES])
        return x
