import torch

print(torch.__version__)

# モデル構築
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, dim_input, dim_hidden_list, dim_output, dropout=0.0):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = GCNConv(dim_input, dim_hidden_list[0])
        self.conv_list = [GCNConv(dim_hidden_list[i], dim_hidden_list[i + 1]) for i in range(len(dim_hidden_list) - 1)]
        self.convl = GCNConv(dim_hidden_list[-1], dim_output)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.convl(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, self.dropout, training=self.training)
        for conv in self.conv_list:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, self.dropout, training=self.training)
        # 出力層は使わず、途中の中間層の出力を返す
        return x
