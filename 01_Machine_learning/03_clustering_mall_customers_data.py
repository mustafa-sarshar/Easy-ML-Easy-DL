# In[] Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_clustering import ClusteringModel

# In[] init the dataset
df_gait = pd.read_csv("datasets/Gait/gait_LeftAnkle.csv", skiprows=12)
X = df_gait.loc[:, ["Gyr_Z", "FreeAcc_U"]]
# y = df_gait.loc[:, -1]
# n_cluster = len(set(y))

# In[] Fit the model
_scaling_method = None
cluster = ClusteringModel(X=X.values, X_labels=X.columns, visualization=True, scaling_method_X=_scaling_method)

# In[] Test the Models for clustering analysis
cluster.estimate_k(clustering_model="KMeans", k_range=(2, 10), method="elbow_yellowbrick_auto", metric="distortion")
cluster.estimate_k(clustering_model="KMeans", k_range=(2, 10), method="elbow_yellowbrick_auto", metric="calinski_harabasz")
cluster.estimate_k(clustering_model="", k_range=(2, 10), method="elbow_kmeans", metric="")
cluster.estimate_k(clustering_model="", k_range=(2, 10), method="dendrogram", metric="")

cluster.settings["kMeans_n_cluster"] = 2
cluster_kMeans = cluster.kMeansClustering_train()

cluster.settings["hierarchy_n_cluster"] = 2
cluster_kMeans = cluster.hierarchicalClustering_train()
