# station2vec

<p align="center">
  <img src="logo.png" />
</p>

This tool provides an implementation of the VGAE (Variational Grph Autoencoder) for computing vector representations of stations.

# model
Based of [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) minimize the loss function below:

```math
loss = \mathbb{E} logp(X|Z) + \mathbb{E}[logp(A|Z)] - KL(q(Z|X, A)||p(Z))
```

where $KL[q(·)||p(·)]$ is the Kullback-Leibler divergence between $q(·)$ and $p(·)$. $A$ is an adjacency matrix. $X$ is a feature matrix. $Z$ is a latent vector.

# 利用データ
以下のデータをを加工して作成（データは全て2023年8月30日取得）。
- 「全国の人流オープンデータ」（国土交通省） https://www.geospatial.jp/ckan/dataset/mlit-1km-fromto
- 「駅データ」（駅データ.jp） https://ekidata.jp/
- 「国土数値情報（地価公示データ）」（国土交通省） https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L01-v3_0.html

# release
From the links below, you can search for similar or dissimilar stations.

https://atsushi-green.github.io/station2vec/

