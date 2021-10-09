class RegressionModel:
    
    def __init__(
            self,
            X=[],
            y=[],
            poly_degree=1,
            X_label="X_label",
            y_label="y_label",
            scaling_method=None,
            visualization=False
    ):
        import numpy as np
        
        self.X = np.array(X)
        self.y = np.array(y)
        self.poly_degree = poly_degree
        self.X_label = X_label
        self.y_label = y_label
        self.scaling_method = scaling_method
        self.visualization = visualization
        self.simpleLinearRegressor = None
        self.polynomialRegressor = None
        self.polynomialLinearRegressor = None
        
    def __repr__(self):
        return f"X: {self.X.shape}, y: {self.y.shape}"

    def simpleLinearRegression_train(self):
        
        from sklearn.linear_model import LinearRegression
        import matplotlib.pyplot as plt     
        
        if self.scaling_method != None:
            print(self.scaling_method)
        
        # Fitting Simple Linear Regression to the Training set        
        model = LinearRegression()
        model.fit(self.X.reshape(-1, 1), self.y.flatten())       
        
        if self.visualization == True:
            # Visualising the Training set results
            plt.scatter(self.X, self.y, color="black")
            plt.plot(self.X, model.predict(self.X.reshape(-1, 1)), color="red")
            plt.title(f"{self.X_label} vs. {self.y_label} (Training set)")
            plt.xlabel(self.X_label)
            plt.ylabel(self.y_label)
            plt.show()
        
        self.simpleLinearRegressor = model
        return model
    
    def simpleLinearRegression_predict(self, y_topred=None):
        # Predicting the Test set results
        if (self.simpleLinearRegressor != None):
            if (y_topred.all):
                return self.simpleLinearRegressor.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def multipleLinearRegression():
        pass
    
    def polynominalRegression_train(self):
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Fitting Polynomial Regression to the dataset
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        poly_reg = PolynomialFeatures(degree=self.poly_degree)
        X_poly = poly_reg.fit_transform(self.X.reshape(-1, 1))
        poly_reg.fit(X_poly, self.y)
        poly_lin_reg = LinearRegression()
        poly_lin_reg.fit(X_poly, self.y)
                
        if self.visualization == True:
            # Visualising the Training set results
            X_grid = np.arange(min(self.X), max(self.X), 0.1).reshape(-1, 1)
            plt.scatter(self.X, self.y, color="black")
            plt.plot(X_grid, poly_lin_reg.predict(poly_reg.fit_transform(X_grid)), color="red")
            plt.title(f"{self.X_label} vs. {self.y_label} (Polynomial Regression)")
            plt.xlabel(self.X_label)
            plt.ylabel(self.y_label)
            plt.show()
        
        self.polynomialRegressor = poly_reg
        self.polynomialLinearRegressor = poly_lin_reg
        
        return poly_reg, poly_lin_reg
    
    def polynominalRegression_predict(self, y_topred=None):
        # Predicting the Test set results
        if (self.polynomialLinearRegressor != None):
            if (y_topred.all):
                return self.polynomialLinearRegressor.predict(self.polynomialRegressor.fit_transform(y_topred))
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def supportVectorRegression(
            X_array = [],
            y_array = [],
            X_label = "X_Label",
            y_label = "y_Label",
            y_topred = 0,
            test_size = 1/3,
            scaling = False,
            visualization = False
    ):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Initializing the variables
        X = X_array
        y = y_array
        
        # Feature Scaling
        if scaling == True:
            from sklearn.preprocessing import StandardScaler
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X = sc_X.fit_transform(X)
            y = sc_y.fit_transform(y)
        
        # Fitting SVR to the dataset
        from sklearn.svm import SVR
        regressor = SVR(kernel="rbf")
        regressor.fit(X, y)
        
        # Predicting a new result
        if scaling == True:
            y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[y_topred]]))))
        else:
            y_pred = regressor.predict(np.array([[y_topred]]))
                
        # Visualising the SVR results (for higher resolution and smoother curve)
        X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X, y, color="red")
        plt.plot(X_grid, regressor.predict(X_grid), color="blue")
        plt.title(f"{X_label} vs. {y_label} (SupportVector Regression)")
        plt.xlabel(X_label)
        plt.ylabel(y_label)
        plt.show()
            
        return y_pred
    
    def decisionTreeRegression(
            X_array = [],
            y_array = [],
            X_label = "X_Label",
            y_label = "y_Label",
            y_topred = 0,
            test_size = 1/3,
            visualization = False
    ):
    
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Initializing the variables
        X = X_array
        y = y_array
        
        # Fitting Decision Tree Regression to the dataset
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state=0)
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
    
    N = 100000
    X = np.random.randn(N) * N - N
    y = X ** 2
    
    plt.clf()
    plt.scatter(X, y)
    plt.show()    
     
    reg = RegressionModel(
        X=X,
        y=y,
        X_label="X_label",
        y_label="y_label",
        visualization = True
    )
    
    # Test the simpleLinearRegression method   
    reg_SLR = reg.simpleLinearRegression_train()
    y_topred = np.array([20, 500, 600]).reshape(-1, 1)
    reg_SLR.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.simpleLinearRegression_predict(y_topred))
    
    
    # Test the polynomialLinearRegressor
    reg.poly_degree = 4
    reg.visualization = True
    reg_PolyReg = reg.polynominalRegression_train()
    y_topred = np.array([20, 500, 600]).reshape(-1, 1)
    reg_SLR.predict(y_topred)
    print(y_topred.flatten(), "-->", reg.polynominalRegression_predict(y_topred))

    