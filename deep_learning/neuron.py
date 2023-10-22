import numpy as np


class Neuron:
    def __init__(self, examples):
        np.random.seed(42)
        # Three weights: one for each feature and one more for the bias.
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.alpha = 0.01
        self.train()

    def sigmoid(self,x) : 
        return 1/(1+np.exp(-x))
    def train(self, learning_rate=0.01, batch_size=10, epochs=200):
        examples = np.array([x["features"] for x in self.examples])
        labels = np.array([x["label"] for x in self.examples])
        ones_column = [[1] for _ in range(len(examples))]
        examples = np.concatenate([examples, ones_column],axis=1)
        for _ in range(epochs) : 
            for i in range(int(len(examples)/batch_size)) : 
                batch = examples[i*batch_size:(i+1)*batch_size]
                batch_labels = labels[i*batch_size:(i+1)*batch_size]
                output = self.sigmoid(batch@self.weights.T)
                #loss = (-labels*np.log(output)-(1-labels)*np.log(1-output)).mean()
                gradients = (batch*(output-batch_labels).reshape(-1,1)).mean(axis=0)
                #gradients = np.dot(batch.T, (output - batch_labels)) / batch_size
                self.weights -= learning_rate*(gradients)

    def predict(self, features):
        features.append(1)
        print(features)
        return self.sigmoid(features@self.weights.T)
