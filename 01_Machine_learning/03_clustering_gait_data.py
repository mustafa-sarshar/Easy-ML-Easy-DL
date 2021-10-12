# In[] Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_clustering import ClusteringModel

# In[] init the dataset
df_breast_cancer = pd.read_csv("datasets/others/breast_cancer.csv")
X = df_breast_cancer.iloc[:, 1:3]
y = df_breast_cancer.iloc[:, -1]
n_cluster = len(set(y))

# In[] Fit the model
_scaling_method = None
cluster = ClusteringModel(X=X.values, X_labels=X.columns, visualization=True, scaling_method_X=_scaling_method)

# In[] Test the Models for clustering analysis
cluster.calculate_wcss(clustering_method="kmeans", n_clusters_max=15, wcss_method="elbow")

cluster.settings["kMeans_n_cluster"] = 4
cluster_kMeans = cluster.kMeansClustering_train()
