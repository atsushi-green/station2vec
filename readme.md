# station2vec
駅の特徴をベクトルで表現する（station2vec）

# model
Variational Graph Auto-Encoders (https://arxiv.org/abs/1611.07308) を元に、
損失を入力特徴ベクトルとの誤差に、内部のGCN層をGraphSAGE (https://arxiv.org/abs/1706.02216) に置き換え

# 利用データ
以下のデータをを加工して作成（データは全て2023年8月30日取得）。
- 「全国の人流オープンデータ」（国土交通省）（https://www.geospatial.jp/ckan/dataset/mlit-1km-fromto）
- 「駅データ」（駅データ.jp）（https://ekidata.jp/）
- 「国土数値情報（地価公示データ）」（国土交通省）（https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-v3_0.html）

# 似ている駅の検索サービス
https://atsushi-green.github.io/station2vec/

