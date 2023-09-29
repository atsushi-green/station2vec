from typing import List, Optional, Tuple

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
def make_station_dataframe(ps: PathSetting, standrize: Optional[bool] = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        weekday_noon_population_list, weekday_night_population_list = [], []
        holiday_noon_population_list, holiday_night_population_list = [], []
        for lon, lat in zip(company_df["経度"].values, company_df["緯度"].values):
            mesh_id = population.search_mesh_id(lon, lat)
            weekday_noon_population_list.append(population.get_population(mesh_id, "noon", "weekday"))
            weekday_night_population_list.append(population.get_population(mesh_id, "night", "weekday"))
            holiday_noon_population_list.append(population.get_population(mesh_id, "noon", "holiday"))
            holiday_night_population_list.append(population.get_population(mesh_id, "night", "holiday"))

        company_df["平日昼人口"] = weekday_noon_population_list
        company_df["平日深夜人口"] = weekday_night_population_list
        company_df["休日昼人口"] = holiday_noon_population_list
        company_df["休日深夜人口"] = holiday_night_population_list
        company_df["平日昼夜人口比"] = np.array(weekday_noon_population_list) / np.array(weekday_night_population_list)
        company_df["休日昼夜人口比"] = np.array(holiday_noon_population_list) / np.array(holiday_night_population_list)
        company_df["平日休日昼人口比"] = np.array(weekday_noon_population_list) / np.array(holiday_noon_population_list)
        company_df["平日休日深夜人口比"] = np.array(weekday_night_population_list) / np.array(holiday_night_population_list)

        # 乗降者数と地価は社局それぞれで標準化
        if standrize:
            company_df = standrize_df(
                company_df,
                ["乗降者数", "地価", "平日昼人口", "平日深夜人口", "休日昼人口", "休日深夜人口", "平日昼夜人口比", "休日昼夜人口比", "平日休日昼人口比", "平日休日深夜人口比"],
            )
        node_df_list.append(company_df)
        edge_df_list.append(company_edge_df)

    station_df = pd.concat(node_df_list)
    edge_df = pd.concat(edge_df_list)

    return station_df, edge_df


def standrize_df(df: pd.DataFrame, col_names: List[str]) -> pd.DataFrame:
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
    fig = plt.figure(figsize=(6.5, 6.5))
    plt.subplots_adjust(right=0.95, left=0.05, bottom=0.05, top=0.95)
    # left=0, right=1, bottom=0, top=1
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
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    if fig_dim == 3:
        save_rotate_movie(fig, ax)
    plt.show()


def save_rotate_movie(fig, ax):
    def animate(i):
        if i < 360:
            # 方位角を変える
            ax.view_init(elev=30.0, azim=91 + i * 2)
        else:
            # 仰角を変える
            ax.view_init(elev=30.0 + i, azim=0)
        return (fig,)

    def init():
        ax.view_init(elev=30.0, azim=0)
        return (fig,)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=40, interval=100, blit=True)
    ani.save("3d-scatter.gif", writer="imagemagick", savefig_kwargs={"transparent": True})
