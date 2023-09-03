from typing import final

import torch
from PathSetting import PathSetting
from pre_post_process import draw_feature, make_station_dataframe, save_features
from StationData import CROSS_ENTROPY_INDEXES, SQUARED_INDEXES, StationData
from torch import nn
from torch_geometric.nn import VGAE
from tqdm import tqdm
from VariationalGraohAutoEncoder import VariationalGraohAutoDecoder, VariationalGraohAutoEncoder

EMBEDDING_DIM = 5
HIDDIN_DIM_LIST: final = [10, 10, 10, 10, 10, 10, 10, 10]


bce_loss_func = nn.BCELoss()


def loss_function(y: torch.Tensor, target: torch.Tensor, model: VGAE) -> torch.Tensor:
    # 二乗誤差(この後クロスエントロピー誤差を加えるので、meanではなくsumを利用しておく)
    sum_loss = torch.sum((y[:, SQUARED_INDEXES] - target[:, SQUARED_INDEXES]) ** 2)

    # クロスエントロピー誤差を加える
    y[:, CROSS_ENTROPY_INDEXES] = torch.clamp(y[:, CROSS_ENTROPY_INDEXES], min=1e-7, max=1 - 1e-7)
    for ce_index in CROSS_ENTROPY_INDEXES:
        sum_loss += bce_loss_func(y[:, ce_index], target[:, ce_index])
    # 平均化して、サイズに依存しないようにして、オートエンコーダーとしての損失を確定させる
    recon_loss = sum_loss / target.shape[0]

    # 負のKLダイバージェンス（VGAEがすでに負のKLを実装してくれている）
    kl_loss = (1 / target.shape[0]) * model.kl_loss()

    return recon_loss + kl_loss


def main():
    ps = PathSetting()
    # データ読み込み
    station_df, edge_df = make_station_dataframe(ps)
    station_df.to_csv("station.csv")  # 今回のデータを保存

    dataset = StationData(station_df, edge_df, standrize=False)
    dataset.print_graph_info()
    data = dataset[0]

    # モデル定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAE(
        VariationalGraohAutoEncoder(
            in_channels=dataset.input_feature_dim,
            hidden_channels_list=HIDDIN_DIM_LIST,
            out_channels=EMBEDDING_DIM,
        ),
        VariationalGraohAutoDecoder(EMBEDDING_DIM, HIDDIN_DIM_LIST[::-1], dataset.input_feature_dim),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 学習
    for epoch in tqdm(range(1, 100001)):
        loss = train(model, optimizer, data)
        if epoch % 100 == 0:
            # train_loss, val_loss, test_loss = test(model, data, loss_function)
            tqdm.write(f"epoch:{epoch}, loss:{loss:.4f}")

    # 学習した埋め込み表現の保存
    model.eval()
    out = model.encode(data.x, data.edge_index)

    save_features(ps, out.detach().cpu().numpy(), station_df["駅名"].values)
    draw_feature(out.detach().cpu().numpy(), station_df["駅名"].values, station_df["地価"].values)


def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    y = model.decode(z, train_data.edge_index)
    # 元論文ではエッジ予測誤差(recon_loss)を損失としているが、
    # 今回はノード特徴量の再構成誤差を損失としたいので、自前で用意する
    # loss = model.recon_loss(z, train_data.edge_index)
    loss = loss_function(y, train_data.x, model)
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
