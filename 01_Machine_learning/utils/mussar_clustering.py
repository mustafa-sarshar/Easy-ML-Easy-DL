class ClusteringModel:
    
    def __init__(
            self,
            X=[],
            X_labels=["X_label_1", "X_label_2"],
            scaling_method_X=None,
            visualization=False,
            settings=dict(
                kMeans_n_cluster=1,
                kMeans_init="k-means++",
                hierarchy_n_cluster=1,
                random_state=0,
                plot_unscaled_data=True,
            )
    ):
        import numpy as np
        
        self.X = np.array(X)
        self.X_labels = X_labels    
        self.scaling_method_X = scaling_method_X
        self.scaler_X = None
        self.visualization = visualization
        self.settings=settings
        self.kMeansClusterer = None
        self.hierarchicalClusterer = None
        
        self.scale_data()
        
    def __repr__(self):
        return f"X: {self.X.shape}"
    
    def get_scalers(self):
        return dict(
            scaler_X=self.scaler_X,
            scaler_method_X=self.scaling_method_X,
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
    
    def calculate_wcss(self, clustering_method="kmeans", n_clusters_max=2, wcss_method="elbow"):
        
        if clustering_method == "kmeans":
            if wcss_method == "elbow":
                import matplotlib.pyplot as plt
                from sklearn.cluster import KMeans
                # Using the elbow method to find the optimal number of clusters            
                wcss = []
                for _n_cluster in range(1, n_clusters_max):
                    kmeans = KMeans(n_clusters=_n_cluster, init=self.settings["kMeans_init"], random_state=self.settings["random_state"])
                    kmeans.fit(self.X)
                    wcss.append(kmeans.inertia_)
                plt.plot(range(1, n_clusters_max), wcss)
                plt.title("The Elbow Method")
                plt.xlabel("Number of clusters")
                plt.ylabel("WCSS")
                plt.show()
            
        if clustering_method == "hierarchy":
            if wcss_method == "dendrogram":
                # Using the dendrogram to find the optimal number of clusters
                import scipy.cluster.hierarchy as sch
                sch.dendrogram(sch.linkage(self.X, method='ward'))
                plt.title("Dendrogram")
                plt.xlabel("Customers")
                plt.ylabel("Euclidean distances")
                plt.show()

    def kMeansClustering(self):
        
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans
        
        # Fitting K-Means to the dataset
        model = KMeans(n_clusters=self.settings["kMeans_n_cluster"], init=self.settings["kMeans_init"], random_state=self.settings["random_state"])
        y_kmeans = model.fit_predict(self.X)
        
        # Cluster with new appropriate names
        colorlist = ['b', 'g', 'r', 'c', 'm']
        for i in range(0, self.settings["kMeans_n_cluster"]):
            plt.scatter(self.X[y_kmeans == i, 0], self.X[y_kmeans == i, 1], s=100, c=colorlist[i], label='Cluster '+str(i+1))
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
        plt.title("K-Means Clustering")
        plt.xlabel(self.X_labels[0])
        plt.ylabel(self.X_labels[1])
        plt.legend()
        plt.show()
        
        self.kMeansClusterer = model
        return model
        
    def hierarchicalClustering(self):
        # Importing the libraries
        import matplotlib.pyplot as plt
        from sklearn.cluster import AgglomerativeClustering
        
       
        model = AgglomerativeClustering(n_clusters=self.settings["hierarchy_n_cluster"], affinity = 'euclidean', linkage = 'ward')
        y_hc = model.fit_predict(self.X)
        
        # Cluster with new appropriate names
        colorlist = ['b', 'g', 'r', 'c', 'm']
        for i in range(0, self.settings["hierarchy_n_cluster"]):
            plt.scatter(self.X[y_hc == i, 0], self.X[y_hc == i, 1], s = 100, c = colorlist[i], label = 'Cluster '+str(i+1))
        plt.title("Hierarchical Clustering")
        plt.xlabel(self.X_labels[0])
        plt.ylabel(self.X_labels[1])
        plt.legend()
        plt.show()
        
        self.hierarchicalClusterer = model
        return model