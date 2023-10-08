from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures

# 与えるノード特徴量
USE_FEATURES = ["地価", "次数", "平日昼人口", "平日深夜人口", "休日昼人口", "休日深夜人口", "平日昼夜人口比", "平日休日昼人口比", "急行"]
SQUARED_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7]
CROSS_ENTROPY_INDEXES = [8]
assert len(USE_FEATURES) == len(SQUARED_INDEXES) + len(CROSS_ENTROPY_INDEXES)

MAX_CONNECT_HOP = 5  # このホップ数より離れた駅とは繋がらない（MAX_CONNECT_HOP ホップまではつなげる）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StationData(InMemoryDataset):
    def __init__(self, station_df: pd.DataFrame, edge_df: pd.DataFrame, standrize=False, transform=None):
        super().__init__(".", transform)

        # 次数(エッジの数)を特徴量として追加する
        station_counter = Counter(edge_df["駅A"].tolist() + edge_df["駅B"].tolist())
        station_df["次数"] = [station_counter[station] for station in station_df["駅名"].values]

        # 乗降者数、地価、次数を特徴量として利用する
        input_data = station_df[USE_FEATURES].values
        self.input_feature_dim = input_data.shape[1]

        input_data = (input_data - input_data.mean(axis=0)) / input_data.std(axis=0) if standrize else input_data
        input_tensor = torch.tensor(input_data, dtype=torch.float).to(device)

        self.station2id = {station: i for i, station in enumerate(station_df["駅名"].values)}
        edge_list, edge_attr = self.calc_graph_distance(station_df, edge_df, self.station2id)

        self.edges = torch.tensor(edge_list, dtype=torch.int64).T.to(device)
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)
        # y(正解データ)をinput_tensorにすることで、オートエンコーダーとして学習させる
        data = Data(
            x=input_tensor,
            edge_index=self.edges,
            y=input_tensor,
            edge_attr=self.edge_attr,
            transform=NormalizeFeatures(),
        ).to(device)

        self.data, self.slices = self.collate([data])
        self.train_mask, self.val_mask, self.test_mask = self.train_val_test_split(data)
        self.data.train_mask = self.train_mask
        self.data.val_mask = self.val_mask
        self.data.test_mask = self.test_mask

    def calc_graph_distance(
        self, station_df: pd.DataFrame, edge_df: pd.DataFrame, station2id: Dict[str, int]
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """路線図上のノードとエッジから、MAX_CONNECT_HOP ホップまでのノード間をエッジで繋ぎ、
        新たに繋いだノード間の距離とホップ数を計算し、エッジ特徴量として保存する

        Args:
            station_df (pd.DataFrame): 駅の特徴量
            edge_df (pd.DataFrame): エッジの情報
            station2id (Dict[str, int]): 駅->idの辞書

        Returns:
            Tuple[List[List[int]], List[List[float]]]: (エッジリスト, エッジ特徴量)
        """

        distance_matrix = np.zeros((len(station2id), len(station2id)), dtype=np.float32)
        hop_matrix = np.zeros((len(station2id), len(station2id)), dtype=np.float32)
        # ノード間グラフ上ので距離を計算する

        # ノードの座標は緯度経度を用いる
        station2pos = {
            station: (lat, lon) for station, lat, lon in zip(station_df["駅名"], station_df["緯度"], station_df["経度"])
        }
        edge_weight_list = []

        for a, b in zip(edge_df["駅A"].values, edge_df["駅B"].values):
            a_lat = station2pos[a][0]
            a_lon = station2pos[a][1]
            b_lat = station2pos[b][0]
            b_lon = station2pos[b][1]
            distance = np.sqrt((a_lat - b_lat) ** 2 + (a_lon - b_lon) ** 2)

            distance_matrix[station2id[a], station2id[b]] = distance
            distance_matrix[station2id[b], station2id[a]] = distance
            edge_weight_list.append([station2id[a], station2id[b], distance])

        # 各ノード間の（路線図上での）最短距離を計算する
        # 空の無向グラフを作成
        G = nx.Graph()
        node_list = list(station2id.values())  # 頂点のリスト
        # 重み付きの枝を加える
        G.add_nodes_from(node_list)  # 頂点の追加#単品可
        G.add_weighted_edges_from(edge_weight_list)  # 重み付き辺の追加
        dijkstra_dict = dict(nx.all_pairs_dijkstra(G))
        # 確認
        # print(a[73][0])  # 73から各ノードへの最短距離
        # print(a[73][1])  # 73から各ノードへの最短経路

        edge_attr = []
        edge_list = []
        for i in range(len(station2id)):
            for j in range(len(station2id)):
                try:
                    distance = dijkstra_dict[i][0][j]  # iを出発点としたときのjまでの距離(0番目の要素)
                    hop = len(dijkstra_dict[i][1][j]) - 1  # iを出発点としたときのjまでのホップ数(1番目の要素のリスト長-1)
                    if hop > MAX_CONNECT_HOP:
                        # 遠すぎるやつは繋がない
                        continue
                    distance_matrix[i, j] = distance
                    hop_matrix[i, j] = hop
                    edge_attr.append([distance, hop])
                    edge_list.append([i, j])
                except KeyError:
                    # 辿り着けないので、エッジをつながない
                    distance_matrix[i, j] = -1
                    hop_matrix[i, j] = -1
        edge_list.extend([i, i] for i in range(len(station2id)))  # 自己ループを追加
        edge_attr.extend([[0, 0]] * len(station2id))  # 自己ループを追加（距離、hop共に0）

        return edge_list, edge_attr

    def train_val_test_split(
        self, data: Data, val_ratio: float = 0.15, test_ratio: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rnd = torch.rand(len(data.x))
        train_mask = [True if (x > val_ratio + test_ratio) else False for x in rnd]
        val_mask = [True if (val_ratio + test_ratio >= x) and (x > test_ratio) else False for x in rnd]
        test_mask = [True if (test_ratio >= x) else False for x in rnd]
        return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)

    def print_graph_info(self) -> None:
        print(f"頂点数: {self.data.num_nodes}")
        print(f"辺数: {self.data.num_edges}")
        print(f"平均次数: {self.data.num_edges / self.data.num_nodes:.2f}")
        print(f"Train用の頂点数: {self.train_mask.sum()}")
        print(f"Val用の頂点数: {self.val_mask.sum()}")
        print(f"Test用の頂点数: {self.test_mask.sum()}")
        print(f"自己ループの有無: {self.data.has_self_loops()}")
        print(f"無向グラフかどうか: {self.data.is_undirected()}")
