# https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-v3_0.html
import json

import numpy as np

# 公示地価
# L01_006が地価
PRICE = "L01_006"
# L01_045が標準地の最寄り駅名
NEAREST_STATION = "L01_045"
# L01_046が最寄り駅までの距離
DISTANCE = "L01_046"
FEATURES = "features"


class LandData:
    def __init__(self, filenames) -> None:
        self.price_list = []
        self.distance_list = []
        self.station_list = []

        for filename in filenames:
            # 京都、大阪、兵庫、東京、神奈川の公示地価データを読み込む
            self.read_geojson(filename)
        # インデックス参照しやすいようにnumpy配列にしておく
        self.price_list = np.array(self.price_list, dtype=np.float32)

    def read_geojson(self, filename):
        with open(filename, "r") as f:
            data = list(json.load(f)[FEATURES])
            price_list = [data[i]["properties"][PRICE] for i in range(len(data))]
            distance_list = [data[i]["properties"][DISTANCE] for i in range(len(data))]
            station_list = [data[i]["properties"][NEAREST_STATION] for i in range(len(data))]
        self.price_list.extend(price_list)
        self.distance_list.extend(distance_list)
        self.station_list.extend(station_list)

    def get_price(self, station_name: str) -> float:
        def findall_index(_list, search_value):
            return [i for i, x in enumerate(_list) if x == search_value]

        station_name = self.convert_near_station(station_name)

        # 複数ある場合は平均値を返す
        index_list = findall_index(self.station_list, station_name)
        return np.average(self.price_list[index_list])

    def convert_near_station(self, station_name: str) -> str:
        """国土数値情報の最寄り駅に存在しないとき、近くにある駅に変換する
        https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-v3_0.html
        9割以上は国土数値情報を検索すれば駅名文字列が完全に一致するが、一致しないものを手動で登録している。

        Args:
            station_name (str): 探したい駅名

        Returns:
            str: station_nameに物理的距離が近しい、国土数値情報の最寄り駅に存在する駅名
        """
        if station_name == "戸越公園":
            # 戸越公園が公示地価にないので、戸越駅の公示地価を返す
            station_name = "戸越"
        elif station_name == "御嶽山":
            # 御嶽山が公示地価にないので、久が原駅の公示地価を返す
            station_name = "久が原"
        elif station_name == "大阪梅田":
            station_name = "梅田"
        elif station_name == "西院":
            station_name = "阪急西院"
        elif station_name == "嵐山":
            station_name = "阪急嵐山"
        elif station_name == "御影":
            station_name = "阪急御影"
        elif station_name == "春日野道":
            station_name = "阪急春日野道"
        elif station_name == "神戸三宮":
            station_name = "三ノ宮"
        elif station_name == "今津":
            station_name = "阪神今津"
        elif station_name == "南方":
            station_name = "崇禅寺"
        elif station_name == "柴島":
            station_name = "崇禅寺"

        return station_name
