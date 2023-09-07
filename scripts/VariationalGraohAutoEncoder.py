from typing import List, Tuple

import torch
from StationData import CROSS_ENTROPY_INDEXES
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ショートカットコネクションを入れたほうがいいかもしれないが、オートエンコーダーの性質上、
# ショートカットコネクションを入れると予測が簡単になりすぎてしまうのではないかという懸念がある。

# Attentionも試したが、学習に時間がかかる上に、精度も悪かったので今回はSAGEConvを使うことにした。
# SAGEConv epoch:100000, loss:2.9652
# GATConv epoch:100000, loss:3.3080


def initial_residual(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """skip connectionを入れるための関数

    Args:
        x (torch.Tensor): 前の層の出力
        h (torch.Tensor): 今の層の出力

    Returns:
        torch.Tensor: skip connectionを入れた後の出力
    """
    x = fit_tensor_size(x=x, out_dim=h.shape[1])
    return x + h


def fit_tensor_size(x: torch.Tensor, out_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """xの各ノードが持つベクトルの次元数をout_dimに合わせる。
    xの方が大きい場合は切り取り、小さい場合は0でパディングする。

    Args:
        x (torch.Tensor): 形を合わせる前のtensor
        out_dim (int): xの2軸目の次元数をこの次元数に合わせる

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 形を合わせた後のtensor
    """
    if x.shape[1] < out_dim:
        # 前の層の出力の次元数の方が小さい時、前の層の出力を0でパディングする
        x = torch.cat([x, torch.zeros((x.shape[0], out_dim - x.shape[1]), dtype=torch.float).to(device)], dim=1)
    elif x.shape[0] > out_dim:
        # 前の層の出力の次元数の方が大きい時、前の層の出力を削って次元数を削る
        x = x[:, :out_dim]
    else:
        # 同じ場合はそのまま
        pass
    return x


class VariationalGraohAutoEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels_list: List[int], out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels_list[0])
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # initial residualするための形揃えるための線形層
        self.initial_residual_list = [
            nn.Linear(in_channels, hidden_channels_list[i]) for i in range(len(hidden_channels_list) - 1)
        ]

        # 最終層だけGCNConvで隣接ノードを全て使う
        self.conv_mu = GCNConv(hidden_channels_list[-1], out_channels)
        self.conv_logstd = GCNConv(hidden_channels_list[-1], out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index).relu()
        h = initial_residual(x, h)
        for conv, res in zip(self.conv_list, self.initial_residual_list):
            # graph conv
            h = conv(h, edge_index).relu()
            # initial residual
            x_ = res(x)
            h = h + x_
            # h = initial_residual(x, h)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


class VariationalGraohAutoDecoder(torch.nn.Module):
    def __init__(self, embedding_channels: int, hidden_channels_list: List[int], out_channels: int):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = SAGEConv(embedding_channels, hidden_channels_list[0])
        self.conv_list = [
            SAGEConv(hidden_channels_list[i], hidden_channels_list[i + 1]) for i in range(len(hidden_channels_list) - 1)
        ]
        # initial residualするための形揃えるための線形層
        self.initial_residual_list = [
            nn.Linear(embedding_channels, hidden_channels)
            for hidden_channels in (hidden_channels_list + [out_channels])
        ]
        # 最終層だけGCNConvで隣接ノードを全て使う
        self.conv_final = GCNConv(hidden_channels_list[-1], out_channels)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(z, edge_index).relu()
        h = initial_residual(z, h)
        for conv, res in zip(self.conv_list, self.initial_residual_list):
            # graph conv
            h = conv(h, edge_index).relu()
            # initial residual
            z_ = res(z)
            h = h + z_

        h = self.conv_final(h, edge_index)
        z_ = self.initial_residual_list[-1](z)
        h = h + z_
        # 0-1をとる変数にはシグモイド関数をかませる
        h[:, CROSS_ENTROPY_INDEXES] = self.sigmoid(h[:, CROSS_ENTROPY_INDEXES])
        return h
