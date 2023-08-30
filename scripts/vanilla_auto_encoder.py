from typing import final

import matplotlib.pyplot as plt
import torch
from GCN import GCN
from PathSetting import PathSetting
from pre_post_process import draw_feature, make_station_dataframe, save_features
from StationData import StationData

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME
HIDDIN_DIM_LIST: final = [20, 20, 20]
WEIGHT_DECAY: final = 5e-4
LR: final = 0.001
DROP_OUT: final = 0.1


def main():
    ps = PathSetting()
    station_df, edge_df = make_station_dataframe(ps)

    # データ読み込み
    dataset = StationData(station_df, edge_df, standrize=False)
    dataset.print_graph_info()
    data = dataset[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習
    model = GCN(
        dim_input=dataset.input_feature_dim,
        dim_hidden_list=HIDDIN_DIM_LIST,
        dim_output=dataset.input_feature_dim,
        dropout=DROP_OUT,
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_function = torch.nn.MSELoss()

    for epoch in range(1, 1001):
        train(model, data, optimizer, loss_function)
        train_loss, val_loss, test_loss = test(model, data, loss_function)
        print(
            f"Epoch: {epoch:04d}",
            f"Train loss: {train_loss:.4f}, " f"Val loss: {val_loss:.4f}, " f"Test loss: {test_loss:.4f}",
        )
    embs = model.get_embeddings(data.x, data.edge_index)

    draw_feature(embs.detach().cpu().numpy(), station_df["駅名"].values, station_df["地価"].values)
    save_features(ps, embs.detach().cpu().numpy(), station_df["駅名"].values)


def train(model, data, optimizer, loss_function):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
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


if __name__ == "__main__":
    main()
