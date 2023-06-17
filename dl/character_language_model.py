import numpy as np
from rnn import RNN

chars = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {key:value for (key,value) in zip(list(chars),range(1,27))}
idx_to_char = {value:key for (value,key) in zip(range(1,27),list(chars))}

#char_to_idx['<S>']=0
#idx_to_char[0] = '<S>'

char_to_idx['\n']=0
idx_to_char[0] = '\n'

with open('dinos.txt') as f : 
    names = f.readlines()
    X = [x.strip().lower() for x in names]

# training loop
num_iterations = 1

rnn_cell = RNN(1,25,27)

for j in range(num_iterations) : 
    j = j%len(X)
    x = X[j]
    y = list(x[1:])+['\n']
    x = [char_to_idx[ch] for ch in list(x)]
    y = np.array([char_to_idx[ch] for ch in y])
    x = np.expand_dims(np.expand_dims(x,axis=0),axis=0)
    y_one_hot_encoded = np.zeros((len(y),27))
    y_one_hot_encoded[np.arange(len(y)),y] = 1
    y_one_hot_encoded = np.expand_dims(y_one_hot_encoded,axis=1)
    y_one_hot_encoded = y_one_hot_encoded.transpose(2,1,0)
    rnn_cell(x,y_one_hot_encoded,epochs=10000,learning_rate=0.01)

    