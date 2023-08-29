import matplotlib.pyplot as plt
import pandas as pd
import torch
from GCN import GCN
from LandData import LandData
from PathSetting import PathSetting
from StationData import StationData

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME


def main():
    ps = PathSetting()
    filenames = ps.get_land_data_filenames()
    land = LandData(filenames)

    # データ読み込み
    # 阪急
    hankyu_station_df = pd.read_csv("../data/hankyu.csv")  # 87駅
    hankyu_station_df["社局"] = ["阪急"] * len(hankyu_station_df)
    hankyu_station_df["地価"] = [land.get_price(station_name) for station_name in hankyu_station_df["駅名"].values]
    hankyu_edge_df = pd.read_csv("../data/hankyu_edge.csv")

    # 東急
    tokyu_station_df = pd.read_csv("../data/tokyu.csv")  # 89駅
    tokyu_station_df["社局"] = ["東急"] * len(tokyu_station_df)
    tokyu_station_df["地価"] = [land.get_price(station_name) for station_name in tokyu_station_df["駅名"].values]
    tokyu_edge_df = pd.read_csv("../data/tokyu_edge.csv")

    hankyu_station_df = standrize(hankyu_station_df)
    tokyu_station_df = standrize(tokyu_station_df)
    station_df = pd.concat([hankyu_station_df, tokyu_station_df])
    edge_df = pd.concat([hankyu_edge_df, tokyu_edge_df])
    # station_df = tokyu_station_df
    # edge_df = tokyu_edge_df

    # station_df = hankyu_station_df
    # edge_df = hankyu_edge_df
    dataset = StationData(station_df, edge_df, land, standrize=False)
    # a, b = zip(*sorted(zip(station_df["乗降者数"], station_df["駅名"])))

    dataset.print_graph_info()
    data = dataset[0]

    # 学習
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(dim_input=2, dim_hidden_list=[3, 3, 2], dim_output=2, dropout=0.2).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    loss_function = torch.nn.MSELoss()

    for epoch in range(1, 1001):
        train(model, data, optimizer, loss_function)
        train_loss, val_loss, test_loss = test(model, data, loss_function)
        print(
            f"Epoch: {epoch:04d}",
            f"Train loss: {train_loss:.4f}, " f"Val loss: {val_loss:.4f}, " f"Test loss: {test_loss:.4f}",
        )
    embs = model.get_embeddings(data.x, data.edge_index)
    print(embs)
    print(embs.shape)
    draw_feature(embs.detach().cpu().numpy(), station_df["駅名"].values)


def train(model, data, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    print("out", out[data.train_mask][0])
    print("y", data.y[data.train_mask][0])
    print()
    print()
    loss = loss_function(out[data.train_mask], data.y[data.train_mask])
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


def standrize(df):
    df["経度"] = (df["経度"] - df["経度"].mean()) / df["経度"].std()
    df["緯度"] = (df["緯度"] - df["緯度"].mean()) / df["緯度"].std()
    df["乗降者数"] = (df["乗降者数"] - df["乗降者数"].mean()) / df["乗降者数"].std()
    df["地価"] = (df["地価"] - df["地価"].mean()) / df["地価"].std()
    return df


def draw_feature(emb, label):
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(right=0.85)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(emb[:, 0], emb[:, 1])
    for label, pos in zip(label, emb):
        ax.text(x=pos[0], y=pos[1], s=label, fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
