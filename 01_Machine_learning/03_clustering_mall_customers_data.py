# In[] Import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_classification import ClassificationModel

# In[] init the dataset
df_breast_cancer = pd.read_csv("datasets/others/breast_cancer.csv")
X = df_breast_cancer.iloc[:, 1:3]
y = df_breast_cancer.iloc[:, -1]

# In[] Fit the model
_scaling_method = None
clss = ClassificationModel(X=X.values, y=y.values, X_label=X.columns, y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
