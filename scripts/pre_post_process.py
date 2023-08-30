from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LandData import LandData
from MeshPopulation import MeshPopulation

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME


# 前処理
def make_station_dataframe(ps):
    filenames = ps.get_land_data_filenames()
    land = LandData(filenames)
    population = MeshPopulation(ps.get_population_filepaths(), ps.get_population_mesh_filepath())

    # 阪急
    # 駅データ
    hankyu_station_df = pd.read_csv(ps.get_station_data_filepath("hankyu"))  # 87駅
    hankyu_station_df["社局"] = ["阪急"] * len(hankyu_station_df)
    hankyu_station_df["地価"] = [land.get_price(station_name) for station_name in hankyu_station_df["駅名"].values]
    # エッジデータ
    hankyu_edge_df = pd.read_csv(ps.get_edge_data_filepath("hankyu"))
    # 昼夜人口
    noon_population_list, night_population_list = [], []
    for lon, lat in zip(hankyu_station_df["経度"].values, hankyu_station_df["緯度"].values):
        mesh_id = population.search_mesh_id(lon, lat)
        noon_population_list.append(population.get_noon_population(mesh_id))
        night_population_list.append(population.get_night_population(mesh_id))
    hankyu_station_df["昼人口"] = noon_population_list
    hankyu_station_df["深夜人口"] = night_population_list
    hankyu_station_df["昼夜人口差"] = np.array(noon_population_list) - np.array(night_population_list)

    # 乗降者数と地価は関西関東それぞれで標準化
    hankyu_station_df = standrize(hankyu_station_df, ["乗降者数", "地価", "昼人口", "深夜人口", "昼夜人口差"])

    # 東急
    # 駅データ
    tokyu_station_df = pd.read_csv(ps.get_station_data_filepath("tokyu"))  # 89駅
    tokyu_station_df["社局"] = ["東急"] * len(tokyu_station_df)
    tokyu_station_df["地価"] = [land.get_price(station_name) for station_name in tokyu_station_df["駅名"].values]
    # エッジデータ
    tokyu_edge_df = pd.read_csv(ps.get_edge_data_filepath("tokyu"))
    # 昼夜人口
    noon_population_list, night_population_list = [], []
    for lon, lat in zip(tokyu_station_df["経度"].values, tokyu_station_df["緯度"].values):
        mesh_id = population.search_mesh_id(lon, lat)
        noon_population_list.append(population.get_noon_population(mesh_id))
        night_population_list.append(population.get_night_population(mesh_id))
    tokyu_station_df["昼人口"] = noon_population_list
    tokyu_station_df["深夜人口"] = night_population_list
    tokyu_station_df["昼夜人口差"] = np.array(noon_population_list) - np.array(night_population_list)

    # 乗降者数と地価は関西関東それぞれで標準化
    tokyu_station_df = standrize(tokyu_station_df, ["乗降者数", "地価", "昼人口", "深夜人口", "昼夜人口差"])

    station_df = pd.concat([hankyu_station_df, tokyu_station_df])
    edge_df = pd.concat([hankyu_edge_df, tokyu_edge_df])

    return station_df, edge_df


def standrize(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    for col_name in col_names:
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
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
