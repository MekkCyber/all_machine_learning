import numpy as np
import random
def sigmoid(z) : 
    return 1/(1+np.exp(-z))

class Layer() : 
    def __init__(self,units,previous_units,activation=None) : 
        self.units = units
        self.previous_units = previous_units
        self.activation = activation
    
    def initialize_weights(self) : 
        self.w = np.random.randn(self.units,self.previous_units)*0.01
        self.b = np.zeros((self.units,1))

    def __call__(self,x) :
        self.initialize_weights()

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
    def __call__(self,x,y,epochs=1,loss_f="binary") : 
        # we add the input to the activations list for later
        m = x.shape[0]
        for i in range(self.num_layers) :
            l = Layer(self.shapes[i+1],self.shapes[i],activation="relu")
            self.layers.append(l)
        
        for i in range(epochs) : 
            input = x.copy()
            # forward propagation
            self.activations = []
            self.activations.append(x)
            for i in range(self.num_layers) :
                input = l(input)
                self.activations.append(input)
                input = input[1]
            output = self.activations[-1][1]
            # compute loss
            if loss_f=="mse" : 
                loss = (1/m)*np.sum(np.square(output-y))
            elif loss_f=="binary" : 
                print(output.shape)
                loss = -(1/m)*np.sum(y*np.log(output)+(1-y)*np.log(1-output))
            # backward propagation 
            grads = {}
            if loss_f == "binary" : 
                dz_l = output - y
                grad[f"dz{self.num_layers}"] = dz_l
                grad[f"dW{self.num_layers}"] = (1/m)*np.matmul(dz_l,self.activations[-2][1].T)
                grad[f"db{self.num_layers}"] = (1/m)*np.sum(dz_l,axis=1,keepdims=True)
            for i in range(self.num_layers-1,0,-1) : 
                grad[f"dz"] = np.matmul(self.layers[i+1].w.T,grad[f"dz{i+1}"])*diff_g(self.activations[i][0])
                grad[f"dW{i}"] = (1/m)*np.matmul(grad[f"dz{i}"],self.activations[i-1][1])
                grad[f"db{i}"] = (1/m)*np.sum(grad[f"dz{i}"],axis=1,keepdims=True)
            print(grads)

dense = Dense(1,[2,3])
dense(np.array([[1,2,3,4,5],[6,7,8,9,10]]),np.array([[1],[0],[0],[1],[0]]))