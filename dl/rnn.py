import numpy as np

def softmax(x,axis=0) : 
    return np.exp(x)/np.sum(np.exp(x),axis=axis)

class Cell : 
    def __init__(self,input_dim,cell_dim,output_dim) : 
        self.input_dim = input_dim
        self.cell_dim = cell_dim
        self.output_dim = output_dim
    
    def initialize(self) : 
        self.Wax = np.random.randn(self.cell_dim,self.input_dim)
        self.Waa = np.random.randn(self.cell_dim,self.cell_dim)
        self.Wya = np.random.randn(self.output_dim,self.cell_dim)
        self.ba = np.zeros((cell_dim,1))
        self.by = np.zeros((output_dim,1))

    def __call__(self,input_step,hidden_state) : 
        # input_step needs to be of dim (input_dim,batch_dim)
        # hidden_state needs to be of dim (cell_dim,batch_dim)
        new_hidden_state = np.tanh(np.matmul(self.Wax,input_step)+np.matmul(self.Waa,hidden_state)+self.ba)
        output = softmax(np.matmul(self.Wya,new_hidden_state)+self.by,axis=0)
        return new_hidden_state,output



