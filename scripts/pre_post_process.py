import matplotlib.pyplot as plt
import pandas as pd
from LandData import LandData

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME


# 前処理
def make_station_dataframe(ps):
    filenames = ps.get_land_data_filenames()
    land = LandData(filenames)

    # 阪急
    hankyu_station_df = pd.read_csv(ps.get_station_data_filepath("hankyu"))  # 87駅
    hankyu_station_df["社局"] = ["阪急"] * len(hankyu_station_df)
    hankyu_station_df["地価"] = [land.get_price(station_name) for station_name in hankyu_station_df["駅名"].values]
    hankyu_edge_df = pd.read_csv(ps.get_edge_data_filepath("hankyu"))

    # 東急
    tokyu_station_df = pd.read_csv(ps.get_station_data_filepath("tokyu"))  # 89駅
    tokyu_station_df["社局"] = ["東急"] * len(tokyu_station_df)
    tokyu_station_df["地価"] = [land.get_price(station_name) for station_name in tokyu_station_df["駅名"].values]
    tokyu_edge_df = pd.read_csv(ps.get_edge_data_filepath("tokyu"))

    hankyu_station_df = standrize(hankyu_station_df)
    tokyu_station_df = standrize(tokyu_station_df)
    station_df = pd.concat([hankyu_station_df, tokyu_station_df])
    edge_df = pd.concat([hankyu_edge_df, tokyu_edge_df])
    return station_df, edge_df


def standrize(df):
    df["経度"] = (df["経度"] - df["経度"].mean()) / df["経度"].std()
    df["緯度"] = (df["緯度"] - df["緯度"].mean()) / df["緯度"].std()
    df["乗降者数"] = (df["乗降者数"] - df["乗降者数"].mean()) / df["乗降者数"].std()
    df["地価"] = (df["地価"] - df["地価"].mean()) / df["地価"].std()
    return df


# 後処理(保存など)
def draw_feature(emb, label, color):
    fig_dim = emb.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.subplots_adjust(right=0.85)
    if fig_dim == 3:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=color, cmap="jet")
    elif fig_dim == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(emb[:, 0], emb[:, 1], c=color, cmap="jet")
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
    if fig_dim == 3:
        save_rotate_movie(fig, ax)
    plt.show()


def save_rotate_movie(fig, ax):
    import matplotlib.animation as animation

    def animate(i):
        if i < 360:
            # 方位角を変える
            ax.view_init(elev=30.0, azim=i)
        else:
            # 仰角を変える
            ax.view_init(elev=30.0 + i, azim=0)
        return (fig,)

    def init():
        ax.view_init(elev=30.0, azim=0)
        return (fig,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=360 * 2, interval=50, blit=True)
    ani.save("3d-scatter.gif", writer="imagemagick")
