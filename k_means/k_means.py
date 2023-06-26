import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()

#sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train,palette="deep")
#plt.xlabel("x")
#plt.ylabel("y")
#plt.show()

def euclidean(x,y):
    return np.sqrt(np.sum((x - y)**2, axis=1))

class KMeans : 
    
    def __init__(self, number_clusters=5) : 
        self.n_clusters = number_clusters
    
    def initialize_simple(self,X) : 
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = [np.random.uniform(min_,max_) for _ in range(self.n_clusters)]

    def initialize_proba(self,X) :    
        '''
        1-If a centroid is initialized far from any groups, it is unlikely to move.
        2-If centroids are initialized too close, they’re unlikely to diverge from one another.
        that is why we use an other algorithm to intialize the centroids : 
        a-Initialize the first centroid as a random selection of one of the data points.
        b-Calculate the sum of the distances between each data point and all the centroids.
        c-Select the next centroid randomly, with a probability proportional to the total distance to the centroids (this way the data point that is far away
        from all the already existing centroids will have the highest probability)
        d-Return to step 2. Repeat until all centroids have been initialized.
        '''
        first_centroid_idx = np.random.choice(range(len(X)))
        self.centroids = [X[first_centroid_idx]]
        for i in range(self.n_clusters-1) : 
            dists = np.sum([euclidean(X,centroid) for centroid in self.centroids],axis=0)
            dists /= np.sum(dists)
            next_centroid_idx = np.random.choice(range(len(X)),p=dists)
            self.centroids.append(X[next_centroid_idx])
    def __call__(self,X,max_num_iterations=100) : 
        self.initialize_proba(X)
        previous_centroids = None
        iteration = 0
        while np.not_equal(self.centroids,previous_centroids).any() and iteration < max_num_iterations : 
            classes = [[] for _ in range(self.n_clusters)]
            for x in X : 
                # x and self.centroids are not the same shape, but we use the effect of broadcasting in python
                distances = euclidean(x,self.centroids)
                centroid = np.argmin(distances)
                classes[centroid].append(x)
            previous_centroids = self.centroids
            self.centroids = [np.mean(x,axis=0) if len(x)>0 else self.centroids[i] for i,x in enumerate(classes)]
            iteration+=1
    
    def evaluate(self, X):
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroid_idxs.append(centroid_idx)        
        return centroid_idxs

centers = 5
X_train, y_train = make_blobs(n_samples=100, centers=5, random_state=42)
X_train = scaler.fit_transform(X_train)
k_means = KMeans()
k_means(X_train)
classification = k_means.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=y_train,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in k_means.centroids],
         [y for _, y in k_means.centroids],
         '+',
         markersize=10,
        )
plt.show()