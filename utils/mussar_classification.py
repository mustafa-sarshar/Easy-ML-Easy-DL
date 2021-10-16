class ClassificationModel:
    
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
                kNN_n_neighbors=5,
                kNN_metric="minkowski",
                kNN_p=2,
                svc_kernel="linear",
                dTree_criterion="entropy",
                rndF_n_estimators=10,
                rndF_criterion="entropy",
                random_state=0,
                plot_unscaled_data=True,
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
        self.logisticClassifier = None
        self.kNNClassifier = None
        self.svClassifier = None
        self.gaussianNBClassifier = None
        self.dTreeClassifier = None
        self.rndFClassifier = None
        
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
            
    def data_visualization(self, model, model_name=""):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.lines import Line2D
        from random import sample
        
        X_dimensions = np.shape(self.X)[1]
        y_features = set(self.y)
        
        colors = sample(list(mcolors.BASE_COLORS)[:-1], len(y_features))
        markers = sample(list(Line2D.markers)[:-4], len(y_features))        
        
        # Visualising the results
        if self.visualization == True:
            if X_dimensions < 2:
                print("Visualization for X: {X_dimensions}D is not possible!")
                
            elif X_dimensions == 2:
                from matplotlib.colors import ListedColormap
                X_set, y_set = self.X, self.y
                X1, X2 = np.meshgrid(
                    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
                )
                plt.contourf(
                    X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha=0.75, cmap=ListedColormap(colors))
                plt.xlim(X1.min(), X1.max())
                plt.ylim(X2.min(), X2.max())
                for i, j in enumerate(np.unique(y_set)):
                    plt.scatter(
                        X_set[y_set == j, 0],
                        X_set[y_set == j, 1],
                        c=ListedColormap(colors)(i),
                        marker=markers[i],
                        label=j
                    )
                
                _x_labels = ", ".join(self.X_label)
                plt.title(f"({model_name})\n{_x_labels} vs. {self.y_label}")
                plt.xlabel(_x_labels)
                plt.ylabel(self.y_label)
                plt.legend()
                plt.show()
                
            elif X_dimensions == 3:
                # Inspired by https://stackoverflow.com/questions/51278752/visualize-2d-3d-decision-surface-in-svm-scikit-learn
                # The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
                # Solve for w3 (z)
                
                zz = lambda x, y: (-model.intercept_[0]-model.coef_[0][0]*x -model.coef_[0][1]*y) / model.coef_[0][2]
                
                tmp = np.linspace(np.min(self.X)-2, np.max(self.X)+2, abs(np.max(self.X)+2 - np.min(self.X)-2)*10)
                xx, yy = np.meshgrid(tmp, tmp)
                
                fig = plt.figure()
                ax  = fig.add_subplot(111, projection="3d")
                for _y in range(len(set(list(self.y)))):
                    ax.plot3D(self.X[self.y==_y, 0], self.X[self.y==_y, 1], self.X[self.y==_y, 2], c=colors[_y], alpha=1, marker=markers[_y])
                ax.plot_surface(xx, yy, zz(xx, yy), alpha=0.7)
                plt.show()
            
            elif X_dimensions > 3:
                print("Visualization for X: {X_dimensions}D is not possible at the moment.!")
        
    def logisticRegression_train(self):
        # Importing the libraries        
        from sklearn.linear_model import LogisticRegression
        
        # Fitting the model
        model = LogisticRegression(random_state=self.settings["random_state"])
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="Logistic Regression Classification")
        self.logisticClassifier = model
        return model
    
    def logisticRegression_predict(self, y_topred=None):
        if (self.logisticClassifier != None):
            if (y_topred.all):
                return self.logisticClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
        
    def kNeighborsClassification_train(self):
        # Importing the libraries        
        from sklearn.neighbors import KNeighborsClassifier
        
        # Fitting the model
        model = KNeighborsClassifier(n_neighbors=self.settings["kNN_n_neighbors"], metric=self.settings["kNN_metric"], p=self.settings["kNN_p"])
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="kNN Classification")
        self.kNNClassifier = model
        return model
    
    def kNeighborsClassification_predict(self, y_topred=None):
        if (self.kNNClassifier != None):
            if (y_topred.all):
                return self.kNNClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def supportVectorClassification_train(self):
        # Importing the libraries        
        from sklearn.svm import SVC
        
        # Fitting the model
        model = SVC(kernel=self.settings["svc_kernel"], random_state=self.settings["random_state"])
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="SVM Classification")
        self.svClassifier = model
        return model
    
    def supportVectorClassification_predict(self, y_topred=None):
        if (self.svClassifier != None):
            if (y_topred.all):
                return self.svClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
        
    def gaussianNaiveBayesClassification_train(self):
        # Importing the libraries        
        from sklearn.naive_bayes import GaussianNB
        
        # Fitting the model
        model = GaussianNB()
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="GaussianNB Classification")
        self.gaussianNBClassifier = model
        return model
    
    def gaussianNaiveBayesClassification_predict(self, y_topred=None):
        if (self.gaussianNBClassifier != None):
            if (y_topred.all):
                return self.gaussianNBClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
    
    def decisionTreeClassification_train(self):
        # Importing the libraries        
        from sklearn.tree import DecisionTreeClassifier
        
        # Fitting the model
        model = DecisionTreeClassifier(criterion=self.settings["dTree_criterion"], random_state=self.settings["random_state"])
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="Decision Tree Classification")
        self.dTreeClassifier = model
        return model
    
    def decisionTreeClassification_predict(self, y_topred=None):
        if (self.dTreeClassifier != None):
            if (y_topred.all):
                return self.dTreeClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
        
    def randomForestClassification_train(self):
        # Importing the libraries        
        from sklearn.ensemble import RandomForestClassifier
        
        # Fitting the model
        model = RandomForestClassifier(n_estimators=self.settings["rndF_n_estimators"], criterion=self.settings["rndF_criterion"], random_state=self.settings["random_state"])
        model.fit(self.X, self.y)        
        
        self.data_visualization(model=model, model_name="Decision Tree Classification")
        self.rndFClassifier = model
        return model
    
    def randomForestClassification_predict(self, y_topred=None):
        if (self.rndFClassifier != None):
            if (y_topred.all):
                return self.rndFClassifier.predict(y_topred)
            else:
                return "y_topred is not defined!"
        else:
            return "Please train the model first!"
        
 
# In[] 
if __name__ == "__main__":
    pass