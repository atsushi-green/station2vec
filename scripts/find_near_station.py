from pathlib import Path

import numpy as np
import pandas as pd
from PathSetting import PathSetting


def main():
    ps = PathSetting()
    vector = stationVector(ps.get_station_vectors_filepath(""))
    k = int(input("上位何件の駅を表示しますか？:"))
    while True:
        station_name = input('駅名を入力してください。終了するには"exit"と入力してください:')
        if station_name == "exit":
            break
        if station_name in vector.station2index:
            vector.get_near_station_global(station_name, k=k)
        else:
            print("存在しない駅です。")


class stationVector:
    def __init__(self, vector_filepath: Path) -> None:
        df = pd.read_csv(vector_filepath)
        self.df = df
        self.station2index = {station_name: i for i, station_name in enumerate(df.columns)}
        # 各駅間のユークリッド距離を保存
        self.distance_matrix = np.zeros((len(df.columns), len(df.columns)), dtype=np.float32)
        for i in range(len(df.columns)):
            for j in range(len(df.columns)):
                self.distance_matrix[i, j] = np.linalg.norm(df.iloc[:, i] - df.iloc[:, j])
        with open("distance_matrix.csv", "w") as f:
            f.write(",".join(df.columns) + "\n")
            for i in range(len(df.columns)):
                f.write(",".join(map(lambda x: f"{x:.4f}", self.distance_matrix[i])) + "\n")

    def get_near_station_global(self, station_name: str, k: int = 5) -> None:
        # 全駅を対象とする
        idx = self.station2index[station_name]
        distances = self.distance_matrix[idx]
        # 自分自身を除いてk件欲しいので、k+1番目までを取得
        min_indices = distances.argsort()[: k + 1]
        # 1番目以降に絞ることで自分自身の駅(station_name)を除く
        print(*list(self.df.iloc[:, min_indices].columns)[1:])

    def get_near_station_local(self, station_name: str, n: int = 5):
        # TODO: 同じ者局の駅のみを対象とする
        pass


if __name__ == "__main__":
    main()
