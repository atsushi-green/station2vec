# 前処理
import pandas as pd
from LandData import LandData


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
