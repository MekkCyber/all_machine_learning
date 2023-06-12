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
        self.ba = np.zeros((self.cell_dim,1))
        self.by = np.zeros((self.output_dim,1))

    def __call__(self,input_step,hidden_state) : 
        # input_step needs to be of dim (input_dim,batch_dim)
        # hidden_state needs to be of dim (cell_dim,batch_dim)
        self.initialize()
        new_hidden_state = np.tanh(np.matmul(self.Wax,input_step)+np.matmul(self.Waa,hidden_state)+self.ba)
        output = softmax(np.matmul(self.Wya,new_hidden_state)+self.by,axis=0)
        return new_hidden_state,output

class RNN : 
    def __init__(self,input_dim,cell_dim,output_dim) : 
        self.input_dim = input_dim
        self.cell_dim = cell_dim
        self.output_dim = output_dim
        self.rnn_cell = Cell(self.input_dim,self.cell_dim,self.output_dim)
    def forward(self,x) : 
        # x the input is of shape (input_dim,batch_dim,sequence_length)
        # the output is of shape (output_dim,batch_dim,sequence_length)
        sequence_length = x.shape[2]
        batch_dim = x.shape[1]
        output = np.zeros((self.output_dim,batch_dim,sequence_length))
        hidden_states = np.zeros((self.cell_dim,batch_dim,sequence_length+1))
        # a referts to the hidden state
        a_0 = np.random.randn(self.cell_dim,batch_dim)
        hidden_states[:,:,0] = a_0.copy()
        a_next = a_0
        for i in range(1,sequence_length+1) : 
            step = x[:,:,i-1]
            a_next, output[:,:,i-1] = self.rnn_cell(step,a_next)
            hidden_states[:,:,i] = a_next
        return hidden_states, output

    def backward(self,hidden_states,output,y,learning_rate=0.001) : 
        # y is of shape (output_dim,batch_dim,sequence_length)
        sequence_length = x.shape[2]
        batch_dim = x.shape[1]
        d_a_next = 0
        d_by, d_Wya, d_Wax, d_Waa, d_ba = 0, 0, 0, 0, 0
        for j in range(sequence_length,0,-1) : 
            d_output = output[:,:,j-1]-y[:,:,j-1]
            d_by += (1/batch_dim)*np.sum(d_output,axis=1,keepdims=True)
            d_Wya += np.matmul(d_output,hidden_states[:,:,j].T)
            d_aj = np.matmul(self.rnn_cell.Wya.T,d_output)+d_a_next
            d_aj_raw = (1-hidden_states[:,:,j]**2)*d_aj
            d_Wax += np.matmul(d_aj_raw,x[:,:,j-1].T)
            d_Waa += np.matmul(d_aj_raw,hidden_states[:,:,j-1].T)
            d_ba += (1/batch_dim)*np.sum(d_aj_raw,axis=1,keepdims=True)
            d_a_next = np.matmul(self.rnn_cell.Waa.T,d_aj_raw)
        for dparam in [d_Wax, d_Waa, d_Wya, d_ba, d_by] :
            np.clip(dparam,-5,5,out=dparam)
        self.rnn_cell.Waa -= learning_rate*d_Waa
        self.rnn_cell.Wax -= learning_rate*d_Wax
        self.rnn_cell.Wya -= learning_rate*d_Wya
        self.rnn_cell.ba -= learning_rate*d_ba
        self.rnn_cell.by -= learning_rate*d_by
    def __call__(self,x,y,learning_rate=0.001,epochs=1) :
        for j in range(epochs) : 
            hidden_states, output = self.forward(x)
            loss = -np.sum(y*np.log(output))
            print(f"the loss in {j} epoch is : {loss}")
            self.backward(hidden_states,output,y,learning_rate)

rnn = RNN(4,5,6)
x = np.array([[[0,1,0,0],[0,1,0,0],[0,1,0,0]],[[0,1,0,0],[0,1,0,0],[0,1,0,0]]])
y = np.array([[[0,1,0,0,0,0],[0,1,0,0,0,0],[0,1,0,0,0,0]]])
x = x.transpose(2,0,1)
y = y.transpose(2,0,1)
rnn(x,y,epochs=1000,learning_rate=0.01)



