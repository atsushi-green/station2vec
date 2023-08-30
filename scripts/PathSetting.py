import os
from pathlib import Path
from typing import List


class PathSetting:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug  # 最小のデータ数で実行確認モード
        self.dir_home = Path.cwd().parent  # scriptsの親ディレクトリ
        self.dir_raw_data = self.dir_home / "data"
        self.dir_population_data = self.dir_raw_data / "population"
        self.dir_output = self.dir_home / "output"
        os.makedirs(self.dir_output, exist_ok=True)

    def get_land_data_filenames(self) -> List[Path]:
        if self.debug:
            return sorted(self.dir_raw_data.glob("**/*.geojson"))[:1]
        else:
            return sorted(self.dir_raw_data.glob("**/*.geojson"))

    def get_station_data_filepath(self, company: str) -> Path:
        # ../data/hankyu.csv などを返す。
        return self.dir_raw_data / f"{company}.csv"

    def get_edge_data_filepath(self, company: str) -> Path:
        # ../data/hankyu.csv などを返す。
        return self.dir_raw_data / f"{company}_edge.csv"

    def get_population_filepaths(self) -> List[Path]:
        # 現在利用可能なデータが2021年までであり、長期休暇などの影響を受けにくい11月のデータを利用する。
        return list(self.dir_population_data.glob("**/2021/11/monthly_mdp_mesh1km.csv"))

    def get_population_mesh_filepath(self) -> Path:
        # 現在の最新のメッシュデータ(2020年)を利用する。
        return self.dir_population_data / "attribute/attribute_mesh1km_2020.csv"

    def get_station_vectors_filepath(self, name: str = "") -> Path:
        if name:
            return self.dir_output / f"station_vectors_{name}.csv"
        else:
            return self.dir_output / "station_vectors.csv"
