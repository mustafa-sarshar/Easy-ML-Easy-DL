import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from utils.mussar_regression import RegressionModel

# In[] init variables
# Create X features
N = 100
X_dimensions = 4
X = np.random.random((N, X_dimensions)) * 20 - 10

# Create y feature
X0_w, X1_w, X2_w, X3_w = 4.5, 12.3, 5.5, 0.5
_intercept = 10

y_1D = _intercept + X0_w*X[:, 0]
y_2D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1]
y_3D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1] + X2_w*X[:, 2]
y_4D = _intercept + X0_w*X[:, 0] + X1_w*X[:, 1] + X2_w*X[:, 2] + X3_w*X[:, 3]

# X = X + np.random.normal(loc=0, scale=1, size=(N, X_dimensions)) # uncomment to add noise to the data

# In[] Plot the data
scaler = MinMaxScaler(feature_range=(1, 10))
_sizes = scaler.fit_transform(y_3D.reshape(-1, 1))
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111, projection='3d')
for _indx in range(np.shape(X)[0]):
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="blue", s=_sizes[_indx][0], alpha=1, marker="s")
    ax.scatter(xs=X[_indx, 0], ys=X[_indx, 1], zs=X[_indx, 2], color="red", s=5, alpha=0.5, marker="o")
plt.show()

# =============================================================================
# ################## Single Dimesional Datasets and Models ####################
# =============================================================================
# In[] Test the Models for 1D datasets
# Init the Regression Class
_scaling_method = None
reg = RegressionModel(X=X[:, 0:1], y=y_1D, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)

reg_LR = reg.linearRegression_train() # Linear Regression

reg.settings["polynomial_degree"] = 2
reg_PolyReg = reg.polynominalRegression_train() # Polynomial Regression

reg.settings["svr_kernel"] = "linear"
reg_svr = reg.supportVectorRegression_train() # Support Vector Regression

reg.settings["random_state"] = 0
reg_decT = reg.decisionTreeRegression_train() # Decision Tree Regression

reg.settings["random_state"] = 0
reg.settings["n_estimators"] = 300
reg_ranF = reg.randomForestRegression_train() # Random Forest Regression

# Test the predictions
y_topred = np.array([[-10], [2], [1200]])

print(f"The fuction is: {_intercept} + ({X0_w})X")
print(f"Given y: {y_topred.flatten()},\texpected y^: [{(_intercept+y_topred[0]*X0_w)[0]} {(_intercept+y_topred[1]*X0_w)[0]} {(_intercept+y_topred[2]*X0_w)[0]}]")
print("Linear Regression")
print(f"y: [{(_intercept+y_topred[0]*X0_w)[0]}, {(_intercept+y_topred[1]*X0_w)[0]}, {(_intercept+y_topred[2]*X0_w)[0]}]", "\ty^:", reg.linearRegression_predict(y_topred))
print("Polynomial Regression")
print(f"y: [{(_intercept+y_topred[0]*X0_w)[0]}, {(_intercept+y_topred[1]*X0_w)[0]}, {(_intercept+y_topred[2]*X0_w)[0]}]", "\ty^:", reg.polynominalRegression_predict(y_topred))
print("Support Vector Regression")
print(f"y: [{(_intercept+y_topred[0]*X0_w)[0]}, {(_intercept+y_topred[1]*X0_w)[0]}, {(_intercept+y_topred[2]*X0_w)[0]}]", "\ty^:", reg.supportVectorRegression_predict(y_topred))
print("Decision Tree Regression")
print(f"y: [{(_intercept+y_topred[0]*X0_w)[0]}, {(_intercept+y_topred[1]*X0_w)[0]}, {(_intercept+y_topred[2]*X0_w)[0]}]", "\ty^:", reg.decisionTreeRegression_predict(y_topred))
print("Random forest Regression")
print(f"y: [{(_intercept+y_topred[0]*X0_w)[0]}, {(_intercept+y_topred[1]*X0_w)[0]}, {(_intercept+y_topred[2]*X0_w)[0]}]", "\ty^:", reg.randomForestRegression_predict(y_topred))

# =============================================================================
# ################### Multi Dimesional Datasets and Models ####################
# =============================================================================
# In[] Test the polynomialLinearRegressor (X: 2D)
_scaling_method = None
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 3D)
_scaling_method = None
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500, 600]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the polynomialLinearRegressor (X: 4D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["polynomial_degree"] = 4
reg_PolyReg = reg.polynominalRegression_train()
y_topred = np.array([[20, 500, 600, 45]])
print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 2D)
_scaling_method = None
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg_LR = reg.linearRegression_train()
y_topred = np.array([[20, 500]])
reg_LR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 3D)
_scaling_method = None
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg_LR = reg.linearRegression_train()
y_topred = np.array([[20, 500, 600]])
reg_LR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the linearRegression method (X: 4D)
_scaling_method = None
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg_LR = reg.linearRegression_train()
y_topred = np.array([[20, 500, 600, 200]])
reg_LR.predict(y_topred)
print(y_topred.flatten(), "-->", reg.linearRegression_predict(y_topred))

# In[] Test the supportVectorRegressor (X: 2D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:2], y=y_2D, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["svr_kernel"] = "rbf"
reg.visualization = True
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  8.43586104]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))

# In[] Test the supportVectorRegressor (X: 3D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:3], y=y_3D, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["svr_kernel"] = "rbf"
reg.visualization = True
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))

# In[] Test the supportVectorRegressor (X: 4D)
_scaling_method = "Standard"
reg = RegressionModel(X=X[:, 0:4], y=y_4D, X_label=["X_label_1", "X_label_2", "X_label_3", "X_label_4"], y_label="y_label", visualization=True, scaling_method_X=_scaling_method, scaling_method_y=_scaling_method)
reg.settings["svr_kernel"] = "rbf"
reg_svr = reg.supportVectorRegression_train()
y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283, 12.334456]])
reg_svr.predict(reg.scaler_X.transform(y_topred))
print(reg.scaler_X.transform(y_topred), "-->", reg.supportVectorRegression_predict(reg.scaler_X.transform(y_topred)))
