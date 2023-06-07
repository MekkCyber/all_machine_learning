import numpy as np
import random
def sigmoid(z) : 
    return 1/(1+np.exp(-z))

class Layer() : 
    def __init__(self,units,previous_units,activation=None) : 
        self.units = units
        self.previous_units = previous_units
        self.activation = activation
        self.initialize_weights()
    
    def initialize_weights(self) : 
        self.w = np.random.randn(self.units,self.previous_units)*0.01
        self.b = np.zeros((self.units,1))

    def __call__(self,x) :
        if self.activation=="relu" :
            return np.matmul(self.w,x)+self.b,np.maximum(np.matmul(self.w,x)+self.b,0)
        elif self.activation=="sigmoid" :
            return np.matmul(self.w,x)+self.b,sigmoid(np.matmul(self.w,x)+self.b)
        else : 
            return np.matmul(self.w,x)+self.b,np.matmul(self.w,x)+self.b

class Dense() : 
    def __init__(self,num_layers,shapes) :
        self.num_layers = num_layers
        # shapes : a list of size num_layers+1 containing the input shape, and the different layer shapes
        self.shapes = shapes
        self.activations = []
        self.layers = []
    def diff_g(self,x) : 
        return sigmoid(x)*(1-sigmoid(x))
    def __call__(self,x,y,epochs=1,loss_f="binary",learning_rate=0.001,epsilon=0.01) : 
        # we add the input to the activations list for later
        m = x.shape[0]
        for i in range(self.num_layers) :
            l = Layer(self.shapes[i+1],self.shapes[i],activation="relu")
            self.layers.append(l)
        for j in range(epochs) : 
            input = x.copy()
            # forward propagation
            self.activations = []
            self.activations.append((x,x))
            for i in range(self.num_layers) :
                input = self.layers[i](input)
                self.activations.append(input)
                input = input[1]
            output = self.activations[-1][1]
            # compute loss
            if loss_f=="mse" : 
                loss = (1/m)*np.sum(np.square(output-y))
            elif loss_f=="binary" : 
                loss = -(1/m)*np.sum(y.T*np.log(output+epsilon)+(1-y).T*np.log(1-output-epsilon))
                accuracy = np.sum((output>=0.5)==y.T) / y.shape[0]
            print(f"the loss for epoch {j} is {loss} and accuracy {accuracy*100}%")
            # backward propagation 
            grad = {}
            if loss_f == "binary" : 
                dz_l = output - y.T
                grad[f"dz{self.num_layers}"] = dz_l
                grad[f"dW{self.num_layers}"] = (1/m)*np.matmul(dz_l,self.activations[-2][1].T)
                grad[f"db{self.num_layers}"] = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            for i in range(self.num_layers-1,0,-1) : 
                grad[f"dz{i}"] = np.matmul(self.layers[i].w.T,grad[f"dz{i+1}"])*self.diff_g(self.activations[i][0])
                grad[f"dW{i}"] = (1/m)*np.matmul(grad[f"dz{i}"],self.activations[i-1][1].T)
                grad[f"db{i}"] = (1/m)*np.sum(grad[f"dz{i}"],axis=1,keepdims=True)
            for i in range(self.num_layers) : 
                self.layers[i].w -= learning_rate*grad[f"dW{i+1}"]
                self.layers[i].b -= learning_rate*grad[f"db{i+1}"]
    def predict(self,x) : 
        for i in range(self.num_layers) :
            x = self.layers[i](x)
            x = x[1]
        return x
dense = Dense(2,[2,3,1])
x = np.array([[1,2,3,4,5],[2,45,65,8,120]])
y = np.array([[1],[0],[0],[1],[0]])
dense(x,y,epochs=1000,learning_rate=0.0005)
print(dense.predict(np.array([[1,2,5],[2,4,44]])))
