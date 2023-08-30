from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LandData import LandData
from MeshPopulation import MeshPopulation
from PathSetting import PathSetting

FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME


# 前処理
def make_station_dataframe(ps: PathSetting) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filenames = ps.get_land_data_filenames()
    land = LandData(filenames)
    population = MeshPopulation(ps.get_population_filepaths(), ps.get_population_mesh_filepath())

    # 阪急
    # 阪急87駅、東急89駅
    node_df_list, edge_df_list = [], []
    for company in ["hankyu", "tokyu"]:
        # 駅データ
        company_df = pd.read_csv(ps.get_station_data_filepath(company))
        company_df["社局"] = [company] * len(company_df)
        # 地価データ
        company_df["地価"] = [land.get_price(station_name) for station_name in company_df["駅名"].values]
        # エッジデータ
        company_edge_df = pd.read_csv(ps.get_edge_data_filepath(company))
        # 昼夜人口
        noon_population_list, night_population_list = [], []
        for lon, lat in zip(company_df["経度"].values, company_df["緯度"].values):
            mesh_id = population.search_mesh_id(lon, lat)
            noon_population_list.append(population.get_noon_population(mesh_id))
            night_population_list.append(population.get_night_population(mesh_id))
        company_df["昼人口"] = noon_population_list
        company_df["深夜人口"] = night_population_list
        company_df["昼夜人口差"] = np.array(noon_population_list) - np.array(night_population_list)
        # 乗降者数と地価は社局それぞれで標準化
        company_df = standrize(company_df, ["乗降者数", "地価", "昼人口", "深夜人口", "昼夜人口差"])
        node_df_list.append(company_df)
        edge_df_list.append(company_edge_df)

    station_df = pd.concat(node_df_list)
    edge_df = pd.concat(edge_df_list)

    return station_df, edge_df


def standrize(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
    for col_name in col_names:
        df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
    return df


def save_features(ps: PathSetting, emb: np.ndarray, station_names: List[str]):
    df = pd.DataFrame(emb, columns=[f"{i:02d}" for i in range(emb.shape[1])])
    # 駅ごとに縦にベクトルを並べるために転置
    df = df.T
    df.columns = station_names
    df.to_csv(ps.get_station_vectors_filepath(), index=False, float_format="%.3f")


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
        # 1次元以下、4次元以上は描画しない
        return

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
