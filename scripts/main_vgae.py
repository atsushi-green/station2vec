from typing import List

import torch
import torch_geometric
from PathSetting import PathSetting
from pre_post_process import draw_feature, make_station_dataframe, save_features
from StationData import CROSS_ENTROPY_INDEXES, SQUARED_INDEXES, StationData
from torch import nn
from torch_geometric.nn import VGAE
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from VariationalGraohAutoEncoder import VariationalGraohAutoDecoder, VariationalGraohAutoEncoder

NUM_EPOCH = 10000
EMBEDDING_DIM = 5
HIDDIN_DIM_LIST: List[int] = [20, 20, 15, 10]
INIT_LEARNING_RATE = 0.01
EPS = 1e-15
bce_loss_func = nn.BCELoss()


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
            edge_attr=data.edge_attr,
        ),
        VariationalGraohAutoDecoder(
            EMBEDDING_DIM, HIDDIN_DIM_LIST[::-1], dataset.input_feature_dim, edge_attr=data.edge_attr
        ),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)

    # 学習
    for epoch in tqdm(range(1, NUM_EPOCH + 1)):
        loss = train(model, optimizer, data)
        if epoch % 100 == 0:
            # train_loss, val_loss, test_loss = test(model, data)
            tqdm.write(f"epoch:{epoch}, loss:{loss:.4f}")

    # 学習した埋め込み表現の保存
    model.eval()
    out = model.encode(data.x, data.edge_index)

    save_features(ps, out.detach().cpu().numpy(), station_df["駅名"].values)
    draw_feature(out.detach().cpu().numpy(), station_df["駅名"].values, station_df["地価"].values)


def train(model: VGAE, optimizer: torch.optim, train_data: torch_geometric.data.data.Data) -> torch.Tensor:
    """モデルによる出力を得て、損失を計算し、パラメータを更新する。

    Args:
        model (VGAE): モデル
        optimizer (torch.optim): オプティマイザー(Adam)
        train_data (torch_geometric.data.data.Data): 学習データセット

    Returns:
        torch.Tensor: 誤差
    """
    model.train()
    optimizer.zero_grad()
    # エンコード->デコードにより、モデルの出力を得る
    z = model.encode(train_data.x, train_data.edge_index)
    y = model.decode(z, train_data.edge_index)
    # 元論文ではエッジ予測誤差(recon_loss)を損失としているが、
    # 今回はノード特徴量の再構成誤差を損失としたいので、自前で用意する
    # loss = model.recon_loss(z, train_data.edge_index)
    loss = loss_function(y, train_data.x, train_data.edge_index, z, model)
    loss.backward()
    optimizer.step()
    return loss


def test(model: VGAE, data: torch_geometric.data.data.Data) -> List[torch.Tensor]:
    # TBD: この関数は未完成
    model.eval()
    out = model(data.x, data.edge_index)
    losses = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        loss = loss_function(out[mask], data.y[mask])
        losses.append(loss)
    return losses


def loss_function(
    y: torch.Tensor, target: torch.Tensor, pos_edge_index: torch.Tensor, z: torch.Tensor, model: VGAE
) -> torch.Tensor:
    """再構成二乗誤差 + 負のKLダイバージェンス+エッジ予測誤差 + エッジ予測誤差を計算する。
    二乗誤差を計算する箇所(SQUARED_INDEXES)は二乗誤差を、
    クロスエントロピー誤差を計算する箇所(CROSS_ENTROPY_INDEXES)はクロスエントロピー誤差を計算する。
    これらの誤差に、負のKLダイバージェンスを計算し、
    さらに、エッジ予測の誤差を加えてVAEの損失を計算する。

    Args:
        y (torch.Tensor): モデルの出力(入力特徴量の再構成)
        target (torch.Tensor): 正解データ(入力特徴量)
        pos_edge_index (torch.Tensor): 繋がっているエッジのインデックス
        z (torch.Tensor): エンコードされたノード特徴量
        model (VGAE): モデル

    Returns:
        torch.Tensor: 誤差
    """
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

    # エッジ予測としての誤差
    pos_loss = -torch.log(model.decoder.edge_pred_forward(z[pos_edge_index[0]], z[pos_edge_index[1]]) + EPS).mean()
    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 - model.decoder.edge_pred_forward(z[neg_edge_index[0]], z[neg_edge_index[1]]) + EPS).mean()
    edge_pred_loss = pos_loss + neg_loss

    return recon_loss + kl_loss + edge_pred_loss


if __name__ == "__main__":
    main()
