from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/data1.csv', encoding='GB2312')
df = df.values
df = df[: ,1:]
KMeans = KMeans(n_clusters=2, max_iter=300, n_init=10).fit(df)
print(KMeans.labels_)
label_pred = KMeans.labels_  # 获取聚类标签
centroids = KMeans.cluster_centers_  # 获取聚类中心
inertia = KMeans.inertia_  # 获取聚类准则的总和
mark = ['^r', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
color = 0
j = 0
for i in label_pred:
    plt.plot([df[j:j+1, 0]], [df[j:j+1, 1]], mark[i], markersize=5)
    j += 1
plt.show()
