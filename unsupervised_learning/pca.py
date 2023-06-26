import numpy as np
import matplotlib.pyplot as plt

class PCA : 
    
    def __init__(self,n_comp=2) : 
        self.n_comp = n_comp

    def __call__(self,X) : 
        # the data X is matrix, where each row is a sample, and each column a feature
        mean = np.mean(X, axis=0)
        X = X-mean
        cov_matrix = np.cov(X.T)
        eig_val, eig_vec = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eig_val)[::-1]
        eig_val = eig_val[sorted_indices]
        eig_vec = eig_vec[:,sorted_indices]
        explained_variance = eig_val / np.sum(eig_val)
        cumulative_variance = np.cumsum(explained_variance)
        components = eig_vec[:,:self.n_comp]
        return cumulative_variance[self.n_comp-1], np.matmul(X,components)

pca = PCA(2)
# create random data
mu = np.array([2,10,3,5])
sigma = np.array([[4,-2,0,1],[-2,3,-1,2],[0,-1,5,10],[1,2,10,4]])
data = np.random.multivariate_normal(mu,sigma,size=(1000,))
'''mu = np.array([10,13])
sigma = np.array([[3.5, -1.8], [-1.8,3.5]])
data = np.random.multivariate_normal(mu, sigma, size=(1000))'''

var, new_data = pca(data)
print("the explained variance is : ",var)
plt.scatter(new_data[:,0],new_data[:,1])
plt.show()