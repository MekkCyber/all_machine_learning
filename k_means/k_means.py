import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

centers = 5
X_train, y_train = make_blobs(n_samples=100, centers=5, random_state=42)
X_train = scaler.fit_transform(X_train)
sns.scatterplot(x=X_train[:,0],y=X_train[:,1],hue=y_train,palette="deep")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

class KMeans : 
    
    def __init__(self, number_clusters=5) : 
        self.n_clusters = number_clusters
    
    def initialize(self,X) : 
        min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        self.centroids = 