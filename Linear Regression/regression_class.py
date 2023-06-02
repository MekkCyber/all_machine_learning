import numpy as np
import matplotlib.pyplot as plt

class LinearRegression() : 
    def __init__(self) :
        self.w = 0
        self.b = 0
        self.m = 0
        self.n = 0
        self.epochs = 10000
        self.step = 1
    def compute_cost(self,x,y,w,b) : 
        return (1/(2*self.m))*np.sum(np.square(np.matmul(w,x)+b-y))
    
    def gradient_descent(self,x,w,b,y,learning_rate=0.001) : 
        dJ_dw = (1/self.m)*(np.matmul(x,(np.matmul(w,x)+b-y).T)).T
        dJ_db = (1/self.m)*(np.sum(np.matmul(w,x)+b-y,axis=1))
        self.w -= dJ_dw*learning_rate
        self.b -= dJ_db*learning_rate
    
    def fit(self,x,y,epochs=100,step=1,learning_rate=0.001) :
        self.m = x.shape[1]
        self.n = x.shape[0]
        self.epochs = epochs
        self.step = step
        self.costs=[]
        self.w = np.random.randn(1,self.n)
        for i in range(epochs) : 
            self.gradient_descent(x,self.w,self.b,y,learning_rate)
            cost = self.compute_cost(x,y,self.w,self.b)
            if i%10==0 : 
                print(f"the {i} iteration : with cost = {cost}")
            if i%step==0 : 
                self.costs.append(cost)
    
    def predict(self,x) : 
        return np.matmul(self.w,x)+self.b
    
    def plot_cost(self) : 
        x = np.arange(0,self.epochs,self.step)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.plot(x,self.costs)
        plt.show()

lr = LinearRegression()
#x = np.array([[5,45,78,74,14],[-5,78,14,25,63],[7,-45,8,12,45],[2,5,99,98,26]])
#y = np.array([[2,8,10,18,156]])
x = np.array([[1,4,5,9,78]])
y = np.array([[2,8,10,18,156]])
lr.fit(x,y,epochs=10,step=1)
lr.plot_cost()
