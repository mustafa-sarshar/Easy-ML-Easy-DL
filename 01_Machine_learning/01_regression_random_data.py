import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_regression import RegressionModel

# In[] init variables
N = 100
X_dimensions = 4
X0_w, X1_w, X2_w, X3_w = 4.5, 12.3, 5.5, 0.5
_intercept = 10
X = np.zeros((N, X_dimensions))
X_sums = np.zeros((1, X_dimensions))
for _indx in range(np.shape(X)[1]):
    X[:, _indx] = (np.random.random(N) * 10 - 10) * (_indx+1)
y_1D = _intercept + X0_w*X[:, 0]
y_2D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1]
y_3D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1] + X2_w*X[:, 2]
y_4D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1] + X2_w*X[:, 2] + X3_w*X[:, 3]
scaler = MinMaxScaler(feature_range=(1, 10))
y_1D_scaled = scaler.fit_transform(y_1D.reshape(-1, 1))
y_2D_scaled = scaler.fit_transform(y_2D.reshape(-1, 1))
y_3D_scaled = scaler.fit_transform(y_3D.reshape(-1, 1))
y_4D_scaled = scaler.fit_transform(y_4D.reshape(-1, 1))

# In[] Plot the data
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, projection='3d')

for _indx in range(np.shape(X)[0]):
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="blue", s=y_3D_scaled[_indx][0], alpha=1, marker="s")
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="red", s=5, alpha=0.5, marker="o")
plt.show()

# In[] Test the linearRegression method (X: 1D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:1], y=y_1D, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([20, 500, 600]).reshape(-1, 1)
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 2D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([[20, 500]])
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 3D)
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([[20, 500, 600]])
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 4D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg_SLR = reg.linearRegression_train()
y_topred = np.array([[20, 500, 600, 200]])
reg_SLR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 1D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:1], y=y_1D, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 2D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 3D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500, 600]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 4D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500, 600, 45]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the supportVectorRegressor (X: 1D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:1], y=y_1D, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["svr_kernel"] = "rbf"
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[20], [40], [1600]])    
reg.scaler_X.inverse_transform(reg_svr.predict(reg.scaler_X.transform(y_topred)))
print(y_topred.flatten(), "-->", reg.supportVectorRegression_predict(y_topred))

# In[] Test the supportVectorRegressor (X: 2D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["svr_kernel"] = "rbf"
reg.visualization = True
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  8.43586104]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))

# In[] Test the supportVectorRegressor (X: 3D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["svr_kernel"] = "rbf"
reg.visualization = True
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))

# In[] Test the supportVectorRegressor (X: 4D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method) # Init the class
reg.settings["svr_kernel"] = "rbf"
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283, 12.334456]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))
