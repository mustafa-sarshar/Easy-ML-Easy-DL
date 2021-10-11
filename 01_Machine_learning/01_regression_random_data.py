import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_regression import RegressionModel

# In[] init variables
N = 100
X_dimensions = 3
X = np.zeros((N, X_dimensions))
for _indx in range(np.shape(X)[1]):
    X[:, _indx] = (np.random.random(N) * 10 - 10) * (_indx+1)
y = (X[:, 0] + X[:, 1] + X[:, 2]) ** 2
scaler = MinMaxScaler(feature_range=(1, 10))
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# In[] Plot the data
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, projection='3d')

for _indx in range(np.shape(X)[0]):
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="blue", s=y_scaled[_indx][0], alpha=1, marker="s")
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="red", s=5, alpha=0.5, marker="o")
plt.show()

# In[] Test the linearRegression method (X: 1D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([20, 500, 600]).reshape(-1, 1)
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 2D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, [0, 2]], y=y, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([[20, 500]])
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 3D)
reg = RegressionModel(X=X, y=y, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X="Standard", scaling_method_y="Standard") # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([[20, 500, 600]])
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 1D)
reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization = True) # Init the class
reg.settings["polynomial_degree"] = 4
reg.visualization = True
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([20, 500, 600]).reshape(-1, 1)
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the supportVectorRegressor (X: 1D)
reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X="Standard", scaling_method_y="Standard") # Init the class
reg.settings["svr_kernel"] = "rbf"
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[20], [40], [1600]])    
reg.scaler_X.inverse_transform(reg_svr.predict(reg.scaler_X.transform(y_topred)))
# print(y_topred.flatten(), "-->", reg.supportVectorRegression_predict(y_topred))

# In[] Test the supportVectorRegressor (X: 3D)
reg = RegressionModel(X=X, y=y, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X="Standard", scaling_method_y="Standard") # Init the class
reg.settings["svr_kernel"] = "rbf"
reg.visualization = True
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283], [ -9.70901658,  -1.15396285, -15.97706972]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
# print(y_topred.flatten(), "-->", reg.supportVectorRegression_predict(y_topred))

