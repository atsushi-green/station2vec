from typing import List, Optional, Tuple

import torch
from StationData import CROSS_ENTROPY_INDEXES
from torch import nn
from torch_geometric.nn import SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attentionも試したが、学習に時間がかかる上に、精度も悪かったので今回はSAGEConvを使うことにした。
# SAGEConv epoch:100000, loss:2.9652
# SAGEConv epoch:100000, loss:3.3080


class VariationalGraohAutoEncoder(torch.nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels_list: List[int], out_channels: int, dropout_rate: Optional[float] = 0.25
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = SAGEConv(in_channels, hidden_channels_list[0])
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # initial residualするための形揃えるための線形層
        self.initial_residual_list = [
            nn.Linear(in_channels, hidden_channels_list[i]) for i in range(1, len(hidden_channels_list))
        ]

        self.conv_mu = SAGEConv(hidden_channels_list[-1], out_channels)
        self.conv_logstd = SAGEConv(hidden_channels_list[-1], out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index).relu()
        for conv, res in zip(self.conv_list, self.initial_residual_list):
            # graph conv
            h = conv(h, edge_index).relu()
            # initial residual
            x_ = res(x)
            h = h + x_
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VariationalGraohAutoDecoder(torch.nn.Module):
    def __init__(
        self,
        embedding_channels: int,
        hidden_channels_list: List[int],
        out_channels: int,
        dropout_rate: Optional[float] = 0.25,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        # 自己符号化器としてのデコーダー
        self.conv1 = SAGEConv(embedding_channels, hidden_channels_list[0])
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # initial residualするための形揃えるための線形層
        self.initial_residual_list = [
            nn.Linear(embedding_channels, hidden_channels)
            for hidden_channels in (hidden_channels_list[1:] + [out_channels])
        ]
        self.conv_final = SAGEConv(hidden_channels_list[-1], out_channels)

        # エッジ予測としてのデコーダー
        # エッジ予測に InnerProductDecoder を使うと、ベクトルが似たノード同士でエッジができやすくなるので、
        # エッジ予測はニューラルネットワークで行う。
        first_layer = nn.Linear(embedding_channels * 2, hidden_channels_list[0])  # 2つのノードの埋め込みをconcatしている
        mid_layer = [
            nn.Linear(hidden_channels_list[i], hidden_channels_list[i + 1])
            for i in range(len(hidden_channels_list) - 1)
        ]
        self.edge_predict_linear = [first_layer] + mid_layer
        self.edge_predict_final = nn.Linear(hidden_channels_list[-1], 1)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 自己符号化器としてのデコーダー
        h = self.conv1(z, edge_index).relu()
        for conv, res in zip(self.conv_list, self.initial_residual_list):
            # graph conv
            h = conv(h, edge_index).relu()
            h = self.dropout(h)
            # initial residual
            z_ = res(z)
            h = h + z_

        h = self.conv_final(h, edge_index)
        z_ = self.initial_residual_list[-1](z)
        h = h + z_
        # 0-1をとる変数にはシグモイド関数をかませる
        h[:, CROSS_ENTROPY_INDEXES] = self.sigmoid(h[:, CROSS_ENTROPY_INDEXES])
        return h

    def edge_pred_forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1の埋め込みを持つノードと、z2の埋め込みを持つノードの間にエッジがある確率を返す。

        Args:
            z1 (torch.Tensor): 埋め込みベクトル
            z2 (torch.Tensor): 埋め込みベクトル

        Returns:
            torch.Tensor: エッジの有無を表す確率
        """
        # エッジ予測としてのデコーダー
        h = torch.cat([z1, z2], dim=1)
        for linear in self.edge_predict_linear:
            h = linear(h).relu()
            h = self.dropout(h)
        h = self.edge_predict_final(h)
        return self.sigmoid(h)
