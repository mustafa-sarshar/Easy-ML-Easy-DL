class RegressionModel:
    
    def __init__(
            self,
            X=[],
            y=[],
            X_label=["X_label_1"],
            y_label="y_label",
            scaling_method_X=None,
            scaling_method_y=None,
            visualization=False,
            settings=dict(
                polynomial_degree=4,
                svr_kernel="rbf",
                plot_unscaled_data=True,
                random_state=0
            )
    ):
        import numpy as np
        
        self.X = np.array(X)
        self.y = np.array(y).flatten()
        self.X_label = X_label
        self.y_label = y_label        
        self.scaling_method_X = scaling_method_X
        self.scaling_method_y = scaling_method_y
        self.scaler_X = None
        self.scaler_y = None
        self.visualization = visualization
        self.settings=settings
        self.linearRegressor = None
        self.polynomialRegressor = None
        self.polynomialLinearRegressor = None
        self.supportVectorRegressor = None
        self.scale_data()
        
    def __repr__(self):
        return f"X: {self.X.shape}, y: {self.y.shape}"
    
    def get_scalers(self):
        return dict(
            scaler_X=self.scaler_X,
            scaler_y=self.scaler_y,
            scaler_method_X=self.scaling_method_X,
            scaler_method_y=self.scaler_method_y
        )
    
    def scale_data(self):
        # Check for scaling
        if self.scaling_method_X != None:
            if (self.scaling_method_X == "Standard"):
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                self.X = sc_X.fit_transform(self.X)
                self.scaler_X = sc_X
            print(f"X scaled via {self.scaling_method_X} scaler!")
        if self.scaling_method_y != None:
            if (self.scaling_method_y == "Standard"):                
                from sklearn.preprocessing import StandardScaler
                sc_y = StandardScaler()
                self.y = sc_y.fit_transform(self.y.reshape(-1, 1))
                self.scaler_y = sc_y
            print(f"y scaled via {self.scaling_method_y} scaler!")        
    
    def linearRegression_train(self):
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt    
        import numpy as np
        
        X_dimensions = np.shape(self.X)[1]
        
        # Fitting Linear Regression to the Training set        
        model = LinearRegression()
        model.fit(self.X, self.y.flatten()) 
        
        if self.visualization == True:
            # if self.settings["plot_unscaled_data"] == False:
            #     X_to_plot = 
            if X_dimensions == 1:
                # Visualising the Training set results
                plt.scatter(self.X, self.y, color="black", label="y")
                plt.plot(self.X, model.predict(self.X.reshape(-1, 1)), color="red", label="fitted line")
                plt.title(f"{self.X_label[0]} vs. {self.y_label} (Training set)")
                plt.xlabel(self.X_label[0])
                plt.ylabel(self.y_label)
                plt.legend()
            
            if X_dimensions == 2:
                # Visualising the Training set results            
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.y[_indx], color="black", s=10, alpha=1, marker="s")
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=model.predict([self.X[_indx, 0:2]]), color="red", s=5, alpha=0.5, marker="o")
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
                
            if X_dimensions >= 3:
                # Scale y for size parameter
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(1, 10))
                y_scaled = scaler.fit_transform(self.y.reshape(-1, 1))
            
                # Visualising the Training set results            
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.X[_indx, 2], color="black", s=y_scaled[_indx][0], alpha=1, marker="s")
                    _size = scaler.transform(model.predict(self.X[0, :].reshape(-1, X_dimensions)).reshape(-1, 1)).flatten()[0]                
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.X[_indx, 2], color="red", s=_size, alpha=0.5, marker="o")
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
            
            plt.show()
        
        self.linearRegressor = model
        return model
    
    def linearRegression_predict(self, y_topred=None):
        # Predicting the Test set results        
        if (self.linearRegressor != None):
            if (y_topred.all):
                return self.linearRegressor.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def polynominalRegression_train(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Fitting Polynomial Regression to the dataset
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        X_dimensions = np.shape(self.X)[1]
        
        model_poly = PolynomialFeatures(degree=self.settings["polynomial_degree"])
        # X_poly = poly_reg.fit_transform(self.X[:, 0].reshape(-1, 1))
        X_poly = model_poly.fit_transform(self.X)
        model_poly.fit(X_poly, self.y)
        model_linear = LinearRegression()
        model_linear.fit(X_poly, self.y)
        
        if self.visualization == True:
            # if self.settings["plot_unscaled_data"] == False:
            #     X_to_plot = 
            if X_dimensions == 1:
                # Visualising the Training set results
                X_grid = np.arange(min(self.X[:, 0]), max(self.X[:, 0]), 0.1).reshape(-1, 1)
                plt.scatter(self.X[:, 0], self.y, color="black", label="y")
                plt.plot(X_grid, model_linear.predict(model_poly.fit_transform(X_grid)), color="red", label="fitted line")
                plt.title(f"{self.X_label[0]} vs. {self.y_label} (Polynomial Regression)")
                plt.xlabel(self.X_label[0])
                plt.ylabel(self.y_label)
                plt.legend()
            
            if X_dimensions == 2:
                # Visualising the Training set results            
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.y[_indx], color="black", s=10, alpha=1, marker="s")
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=model_linear.predict(model_poly.fit_transform(self.X[0, :].reshape(-1, X_dimensions))), color="red", s=5, alpha=0.5, marker="o")
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
                
            if X_dimensions >= 3:
                # Scale y for size parameter
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(1, 10))
                y_scaled = scaler.fit_transform(self.y.reshape(-1, 1))
            
                # Visualising the Training set results            
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.X[_indx, 2], color="black", s=y_scaled[_indx][0], alpha=1, marker="s")
                    _size = model_linear.predict(model_poly.fit_transform(self.X[0, :].reshape(-1, X_dimensions)))
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.X[_indx, 2], color="red", s=scaler.transform(_size.reshape(-1, 1)), alpha=0.5, marker="o")
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
            
            plt.show()
        
        self.polynomialRegressor = model_poly
        self.polynomialLinearRegressor = model_linear
        
        return model_poly, model_linear
    
    def polynominalRegression_predict(self, y_topred=None):
        # Predicting the Test set results
        if (self.polynomialLinearRegressor != None):
            if (y_topred.all):
                return self.polynomialLinearRegressor.predict(self.polynomialRegressor.fit_transform(y_topred))
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def supportVectorRegression_train(self):
        import numpy as np
        import matplotlib.pyplot as plt
        
        X_dimensions = np.shape(self.X)[1]
        # Kernels: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf'
        # Check for scaling
        if self.scaling_method_X != None:     
            if (self.scaling_method_X == "Standard"):
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                self.X = sc_X.fit_transform(self.X)
                self.scaler_X = sc_X
        if self.scaling_method_y != None:
            if (self.scaling_method_y == "Standard"):                
                from sklearn.preprocessing import StandardScaler
                sc_y = StandardScaler()
                self.y = sc_y.fit_transform(self.y.reshape(-1, 1))
                self.scaler_y = sc_y
        
        # Fitting SVR to the dataset
        from sklearn.svm import SVR
        
        model = SVR(kernel=self.settings["svr_kernel"])
        model.fit(self.X, self.y.flatten())
             
        if self.visualization == True:
            
            if X_dimensions == 1:
                # Visualising the Training set results
                X_grid = np.arange(min(self.scaler_X.inverse_transform(self.X[:, 0])), max(self.scaler_X.inverse_transform(self.X[:, 0])), 0.01).reshape((-1, 1)) # choice of 0.01 instead of 0.1 step because the data is feature scaled
                plt.clf()
                plt.scatter(self.scaler_X.inverse_transform(self.X[:, 0]), self.scaler_y.inverse_transform(self.y), color="black", label="y")
                plt.plot(X_grid, self.scaler_y.inverse_transform(model.predict(self.scaler_X.inverse_transform(X_grid))), color="red", label="fitted line")
                plt.title(f"{self.X_label[0]} vs. {self.y_label} (SupportVector Regression)")
                plt.xlabel(self.X_label)
                plt.ylabel(self.y_label)
                plt.legend()
                plt.show()      
            
            elif X_dimensions == 2:
                # scale y for size parameter
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(1, 10))
                y_scaled = scaler.fit_transform(self.y.reshape(-1, 1))
            
                # Visualising the Training set results            
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=self.y[_indx], color="black", s=10, alpha=1, marker="s")
                    ax.scatter(xs=self.X[_indx, 0], ys=self.X[_indx, 1], zs=model.predict(self.X[0, :].reshape(-1, X_dimensions)), color="red", s=5, alpha=0.5, marker="o")
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
                
            elif X_dimensions >= 3:
                # scale y for size parameter
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(1, 10))
                y_scaled = scaler.fit_transform(self.y.reshape(-1, 1))
            
                # Visualising the Training set results
                fig = plt.figure()
                plt.clf()
                ax = fig.add_subplot(111, projection="3d")
                for _indx in range(np.shape(self.X)[0]):
                    _point = self.scaler_X.inverse_transform(self.X[_indx, :])
                    _size = model.predict(self.scaler_X.transform(_point.reshape(1, -1)))
                    _size_scaled_minmax = scaler.transform(_size.reshape(-1, 1))[0]
                    ax.scatter(
                        xs=_point[0],
                        ys=_point[1],
                        zs=_point[2],
                        s=y_scaled[_indx],
                        color="black", alpha=1, marker="s"
                    )
                    ax.scatter(
                        xs=_point[0],
                        ys=_point[1],
                        zs=_point[2],
                        s=_size_scaled_minmax,
                        color="red", alpha=0.5, marker="o"
                    )
                _x_labels = ", ".join(self.X_label)
                plt.title(f"{_x_labels} vs. {self.y_label} (Training set)")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend(["y", "y_pred"])
                plt.show()
        
        self.supportVectorRegressor = model
        return model
    
    def supportVectorRegression_predict(self, y_topred=None):
        # Predicting the Test set results
        if (self.supportVectorRegressor != None):
            if (y_topred.all):
                return self.supportVectorRegressor.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
        
    def decisionTreeRegression(self):
    
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Fitting Decision Tree Regression to the dataset
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=self.settings["random_state"])
        regressor.fit(X, y)
        
        # Predicting a new result
        y_pred = regressor.predict(np.array([[y_topred]]))
        
        # Visualising the Decision Tree Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color="red")
        plt.plot(X_grid, regressor.predict(X_grid), color="blue")
        plt.title(f"{X_label} vs. {y_label} (DecisionTree Regression)")
        plt.xlabel(X_label)
        plt.ylabel(y_label)
        plt.show()
        
        return y_pred
    
    def randomForestRegression(
            X_array = [],
            y_array = [],
            X_label = "X_Label",
            y_label = "y_Label",
            y_topred = 0,
            n_estimators = 300,
            test_size = 1/3,
            visualization = False
    ):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Initializing the variables
        X = X_array
        y = y_array
        
        # Fitting Random Forest Regression to the dataset
        from sklearn.ensemble import RandomForestRegressor
        #regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
        # For a better prediction n_estimators is set to 300
        regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=0)
        regressor.fit(X, y)
        
        # Predicting a new result
        y_pred = regressor.predict(np.array([[y_topred]]))
        
        # Visualising the Random Forest Regression results (higher resolution)
        X_grid = np.arange(min(X), max(X), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color="red")
        plt.plot(X_grid, regressor.predict(X_grid), color="blue")
        plt.title(f"{X_label} vs. {y_label} (RandomForest Regression)")
        plt.xlabel(X_label)
        plt.ylabel(y_label)
        plt.show()
        
        return y_pred

# In[]
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    
    # In[] init variables
    N = 10
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

    # In[] Test the simpleLinearRegression method (X: 1D)
    reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization = True) # Init the class
    reg.visualization = True
    reg_SLR = reg.simpleLinearRegression_train()
    y_topred = np.array([20, 500, 600]).reshape(-1, 1)
    # reg_SLR.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.simpleLinearRegression_predict(y_topred))
    
    # In[] Test the multipleLinearRegression method (X: 2D)
    reg = RegressionModel(X=X, y=y, X_label=["X_label_1", "X_label_2"], y_label="y_label", visualization = True) # Init the class
    reg.visualization = False
    reg.scaling_method_X = "Standard"
    reg_MLR = reg.multipleLinearRegression_train()
    y_topred = np.array([[20, 30]])
    # reg_MLR.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.multipleLinearRegression_predict(y_topred))
    
    # In[] Test the multipleLinearRegression method (X: 3D)
    reg = RegressionModel(X=X, y=y, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization = True) # Init the class
    reg.visualization = True
    reg.scaling_method_X = "Standard"
    reg_MLR = reg.multipleLinearRegression_train()
    y_topred = np.array([[20, 30, 40]])
    # reg_MLR.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.multipleLinearRegression_predict(y_topred))
    
    # In[] Test the polynomialLinearRegressor (X: 1D)
    reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization = True) # Init the class
    reg.settings["polynomial_degree"] = 4
    reg.visualization = True
    reg_PolyReg = reg.polynominalRegression_train()
    y_topred = np.array([20, 500, 600]).reshape(-1, 1)
    # reg_PolyReg.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))
    
    # In[] Test the supportVectorRegressor (X: 1D)
    reg = RegressionModel(X=X[:, 0:1], y=y, X_label=["X_label_1"], y_label="y_label", visualization=True, scaling_method_X="Standard", scaling_method_y="Standard") # Init the class
    reg.settings["svr_kernel"] = "rbf"
    reg_svr = reg.supportVectorRegression_train()
    y_topred = np.array([[20], [40], [1600]])    
    reg.scaler_X.inverse_transform(reg_svr.predict(reg.scaler_X.transform(y_topred)))
    # print(y_topred.flatten(), "-->", reg.supportVectorRegression_predict(y_topred))
    
    # In[] Test the supportVectorRegressor (X: 3D)
    reg = RegressionModel(X=X, y=y, X_label=["X_label_1", "X_label_2", "X_label_3"], y_label="y_label", visualization = True) # Init the class
    reg.settings["svr_kernel"] = "rbf"
    reg.visualization = True
    reg.scaling_method_X = "Standard"
    reg_svr = reg.supportVectorRegression_train()
    y_topred = np.array([[ -5.02763379,  -8.43586104,  -5.45849283], [ -9.70901658,  -1.15396285, -15.97706972]])
    reg_svr.predict(reg.scaler_X.transform(y_topred))
    # print(y_topred.flatten(), "-->", reg.supportVectorRegression_predict(y_topred))

    