from pathlib import Path
from typing import List


class PathSetting:
    def __init__(self, peft_name: str = "", debug: bool = False) -> None:
        self.debug = debug  # 最小のデータ数で実行確認モード
        self.dir_home = Path.cwd().parent  # scriptsの親ディレクトリ
        self.dir_raw_data = self.dir_home / "data"
        # self.dir_dataset = self.dir_home / "dataset"

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
