# 全国の人流オープンデータ
# 出典：「全国の人流オープンデータ」（国土交通省）（https://www.geospatial.jp/ckan/dataset/mlit-1km-fromto
# 列の定義
# 1. メッシュID
# 2. 都道府県コード
# 3. 市町村コード
# 4. 集計期間（年別）: ex) “2020”
# 5. 集計期間（月別）: ”01” – “12”
# 6. 集計期間（平休日）: “0”:休日 “1”:平日 “2”:全日
# 7. 集計期間（時間帯）: “0”:昼 “1”:深夜 “2”:終日
# 8. 滞在人口(平均): ”10”以上 10人未満は出力しない

# 昼:11時台〜14時台の平均
# 深夜:1時台〜4時台の平均
# 終日:0時台〜23時台の平均


from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

DAY_FLAG = 1  # 平日
TIME_ZONE_NOON = 0  # 昼
TIME_ZONE_NIGHT = 1  # 深夜


class MeshPopulation:
    def __init__(self, populatiopn_filepaths: List[Path], mesh_filepath: Path) -> None:
        # 人口データの読み込み
        df_list = []
        for populatiopn_filepath in populatiopn_filepaths:
            df = pd.read_csv(populatiopn_filepath, dtype={"mesh1kmid": str})
            df = df.set_index("mesh1kmid", drop=False)
            df_list.append(df)
        population_df = pd.concat(df_list)

        # 平日のみ抽出
        population_df = population_df[population_df["dayflag"] == DAY_FLAG]
        # 昼と深夜をそれぞれ抽出
        self.noon_population_df = population_df[population_df["timezone"] == TIME_ZONE_NOON]
        self.night_population_df = population_df[population_df["timezone"] == TIME_ZONE_NIGHT]

        # 全国のメッシュ情報を読み込む
        self.mesh_df = pd.read_csv(mesh_filepath, dtype={"mesh1kmid": str})

    def search_mesh_id(self, lon: float, lat: float) -> str:
        """経度緯度から、最も近いメッシュのIDを返す。

        Args:
            lon (float): 経度
            lat (float): 緯度

        Returns:
            str: 与えられた経度緯度から最も近いメッシュのID
        """
        # 全てのメッシュに対して、経度緯度の二乗誤差を計算する
        squared_errors = (self.mesh_df["lon_center"].values - lon) ** 2 + (self.mesh_df["lat_center"].values - lat) ** 2
        # 最も小さいものを選ぶ
        min_idx = np.argmin(squared_errors)
        # メッシュIDを返す
        return self.mesh_df.iloc[min_idx]["mesh1kmid"]

    def get_noon_population(self, mesh_id: str) -> float:
        """与えられたメッシュID地点の昼の人口を返す

        Args:
            mesh_id (str): 全国のメッシュID

        Returns:
            float: 昼人口
        """
        # メッシュIDから人口を返す
        return self.noon_population_df.loc[mesh_id]["population"]

    def get_night_population(self, mesh_id: str) -> float:
        """与えられたメッシュID地点の深夜の人口を返す

        Args:
            mesh_id (str): 全国のメッシュID

        Returns:
            float: 深夜人口
        """

        # メッシュIDから人口を返す
        return self.night_population_df.loc[mesh_id]["population"]
