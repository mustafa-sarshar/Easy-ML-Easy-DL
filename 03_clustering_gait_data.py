# In[] Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_clustering import ClusteringModel

# In[] init the dataset
_filename = "gait_LeftAnkle.csv"
df_breast_cancer = pd.read_csv(f"datasets/Gait/{_filename}", skiprows=12)
X = df_breast_cancer.loc[:, ["Gyr_Z", "FreeAcc_U"]]

# In[] Fit the model
_scaling_method = None
cluster = ClusteringModel(X=X.values, X_labels=X.columns, visualization=True, scaling_method_X=_scaling_method)

# In[] Test the Models for clustering analysis
cluster.calculate_wcss(clustering_model="KMeans", k_range=(2, 10), wcss_method="elbow_yellowbrick_auto", metric="distortion")
cluster.calculate_wcss(clustering_model="KMeans", k_range=(2, 10), wcss_method="elbow_yellowbrick_auto", metric="calinski_harabasz")
cluster.calculate_wcss(clustering_model="", k_range=(2, 10), wcss_method="elbow_kmeans", metric="")
cluster.calculate_wcss(clustering_model="", k_range=(2, 10), wcss_method="dendrogram", metric="")

cluster.settings["kMeans_n_cluster"] = 3
cluster_kMeans = cluster.kMeansClustering_train()

cluster.settings["hierarchy_n_cluster"] = 3
cluster_kMeans = cluster.hierarchicalClustering_train()
