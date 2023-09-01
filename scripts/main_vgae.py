from typing import final

import torch
from PathSetting import PathSetting
from pre_post_process import draw_feature, make_station_dataframe, save_features
from StationData import CROSS_ENTROPY_INDEXES, SQUARED_INDEXES, StationData
from torch import nn
from torch_geometric.nn import VGAE
from tqdm import tqdm
from VariationalGraohAutoEncoder import VariationalGraohAutoEncoder

EMBEDDING_DIM: final = 10
HIDDIN_DIM_LIST: final = [10, 10, 10]


def main():
    ps = PathSetting()
    # データ読み込み
    station_df, edge_df = make_station_dataframe(ps)
    dataset = StationData(station_df, edge_df, standrize=False)
    dataset.print_graph_info()
    data = dataset[0]

    # モデル定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAE(
        VariationalGraohAutoEncoder(
            in_channels=dataset.input_feature_dim,
            hidden_channels_list=HIDDIN_DIM_LIST,
            out_channels=dataset.input_feature_dim,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 学習
    for _ in tqdm(range(1, 1001)):
        train(model, optimizer, data)
    model.eval()

    out = model.encode(data.x, data.edge_index)

    station_df.to_csv("station.csv")

    save_features(ps, out.detach().cpu().numpy(), station_df["駅名"].values)
    draw_feature(out.detach().cpu().numpy(), station_df["駅名"].values, station_df["地価"].values)


def loss_function(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    loss = nn.CrossEntropyLoss()
    squared_loss = torch.mean((z[SQUARED_INDEXES] - x[SQUARED_INDEXES]) ** 2)
    cross_entropy_loss = loss(z[CROSS_ENTROPY_INDEXES], x[CROSS_ENTROPY_INDEXES])
    return squared_loss + cross_entropy_loss


def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    # 元論文ではエッジ予測誤差(recon_loss)を損失としているが、今回はノード特徴量の再構成誤差を損失とする
    # loss = model.recon_loss(z, train_data.edge_index)
    loss = loss_function(z, train_data.x)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss


def test(model, data, loss_function):
    model.eval()
    out = model(data.x, data.edge_index)
    losses = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        loss = loss_function(out[mask], data.y[mask])
        losses.append(loss)
    return losses


if __name__ == "__main__":
    main()
