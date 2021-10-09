def kMeansClustering(X_array = [],
                X1_label = 'X1_Label',
                X2_label = 'X2_Label',
                y_topred = [],
                test_size = 1/3,
                prestep = True,
                poststep = False,
                n_clusters = 1,
                visualization = False):
    # Importing the libraries
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    
    # Initializing the variables
    X = X_array
    
    if prestep == True:
        # Using the elbow method to find the optimal number of clusters
        
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()
    
    if poststep == True:
        # Fitting K-Means to the dataset
        kmeans = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 0)
        y_kmeans = kmeans.fit_predict(X)
        #y_pred = kmeans.fit_predict([[40, 20]])
        
        # Cluster with new appropriate names
        colorlist = ['b', 'g', 'r', 'c', 'm']
        for i in range(0, n_clusters):
            plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = colorlist[i], label = 'Cluster '+str(i+1))
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
        plt.title('K-Means Clustering')
        plt.xlabel(X1_label)
        plt.ylabel(X2_label)
        plt.legend()
        plt.show()
    
    #return y_pred
    return
        
def hierarchicalClustering(X_array = [],
                            X1_label = 'X1_Label',
                            X2_label = 'X2_Label',
                            y_topred = [],
                            test_size = 1/3,
                            prestep = True,
                            poststep = False,
                            n_clusters = 1,
                            visualization = False):
    # Importing the libraries
    import matplotlib.pyplot as plt
    
    # Initializing the variables
    X = X_array
    
    if prestep == True:
        # Using the dendrogram to find the optimal number of clusters
        import scipy.cluster.hierarchy as sch
        sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.xlabel('Customers')
        plt.ylabel('Euclidean distances')
        plt.show()
    
    if poststep == True:
        # Fitting Hierarchical Clustering to the dataset
        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
        y_hc = hc.fit_predict(X)
        
        # Cluster with new appropriate names
        colorlist = ['b', 'g', 'r', 'c', 'm']
        for i in range(0, n_clusters):
            plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = colorlist[i], label = 'Cluster '+str(i+1))
        plt.title('Hierarchical Clustering')
        plt.xlabel(X1_label)
        plt.ylabel(X2_label)
        plt.legend()
        plt.show()
    
    #return y_pred
    return