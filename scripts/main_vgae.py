from typing import final

import matplotlib.pyplot as plt
import torch
from PathSetting import PathSetting
from preprocess import make_station_dataframe
from StationData import StationData
from torch_geometric.nn import VGAE
from VariationalGraohAutoEncoder import VariationalGraohAutoEncoder

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME
EMBEDDING_DIM: final = 2


def main():
    ps = PathSetting()
    station_df, edge_df = make_station_dataframe(ps)

    # データ読み込み
    dataset = StationData(station_df, edge_df, standrize=False)
    dataset.print_graph_info()
    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGAE(
        VariationalGraohAutoEncoder(
            in_channels=dataset.input_feature_dim, hidden_channels_list=[4], out_channels=EMBEDDING_DIM
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 1001):
        train(model, optimizer, data)
    model.eval()
    mu, std = model(data.x, data.edge_index)

    out = model.encode(data.x, data.edge_index)

    print(mu[:10])
    print(out[:10])
    station_df.to_csv("station.csv")

    draw_feature(out.detach().cpu().numpy(), station_df["駅名"].values, station_df["地価"].values)


def train(model, optimizer, train_data):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.edge_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(model, data, loss_function):
    model.eval()
    out = model(data.x, data.edge_index)
    losses = []
    for _, mask in data("train_mask", "val_mask", "test_mask"):
        loss = loss_function(out[mask], data.y[mask])
        losses.append(loss)
    return losses


def draw_feature(emb, label, color):
    fig_dim = emb.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(right=0.85)
    if fig_dim == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=color, cmap="Reds", edgecolor="r")
    elif fig_dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap="Reds")
    else:
        raise ValueError

    for label, pos in zip(label, emb):
        if label == "渋谷":
            # グラフ上から渋谷を探すために座標を表示
            print(pos)
        if fig_dim == 3:
            ax.text(x=pos[0], y=pos[1], z=pos[2], s=label, fontsize=9)
        elif fig_dim == 2:
            ax.text(x=pos[0], y=pos[1], s=label, fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
