# In[] Import libs
import numpy as np
import pandas as pd
from utils.mussar_classification import ClassificationModel

# In[] init the dataset
df_breast_cancer = pd.read_csv("datasets/others/breast_cancer.csv")
X = df_breast_cancer.iloc[:, 1:3]
y = df_breast_cancer.iloc[:, -1]

# In[] Fit the model
_scaling_method = None
clss = ClassificationModel(X=X.values, y=y.values, X_label=X.columns, y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)

# In[] Test the Models for 2D classification
clss_logReg = clss.logisticRegression_train()
clss_kNN = clss.kNeighborsClassification_train()
clss.settings["svc_kernel"] = "linear"
clss_svc_linear = clss.supportVectorClassification_train()
clss.settings["svc_kernel"] = "rbf"
clss_svc_rbf = clss.supportVectorClassification_train()
clss_gaussNB = clss.gaussianNaiveBayesClassification_train()
clss_dTree = clss.decisionTreeClassification_train()
clss.settings["rndF_n_estimators"] = 100
clss_rndF = clss.randomForestClassification_train()

y_topred = np.array([[4, 1]])
print(y_topred.flatten(), "-->", clss.logisticRegression_predict(y_topred))
print(y_topred.flatten(), "-->", clss.kNeighborsClassification_predict(y_topred))
print(y_topred.flatten(), "-->", clss.supportVectorClassification_predict(y_topred))
print(y_topred.flatten(), "-->", clss.gaussianNaiveBayesClassification_predict(y_topred))
print(y_topred.flatten(), "-->", clss.decisionTreeClassification_predict(y_topred))
print(y_topred.flatten(), "-->", clss.randomForestClassification_predict(y_topred))

