import numpy as np

from utils.preprocessing import data_loader
from utils import activations
class MLP():
    def __init__(self, arch):
        leng = len(arch)
        self.W = [None] * leng
        self.b = [None] * leng
        self.act = [None] * leng
        self.leng = leng
        for l in range(leng):
            self.W[l] = np.random.randn(arch[l][0], arch[l][1])
            self.b[l] = np.random.rand(1, arch[l][1])
            self.act[l] = arch[l][2]

    def __len__(self):
        return self.leng

    def summary(self):
        print('Model Summary: ', '==============', sep='\n')
        print(36*'-')
        for l in range(len(self)):
            print('| Layer #{} => '.format(l), end='')
            print('W' + str(l) + ': ' + str(self.W[l].shape), end=' ')
            print('and b' + str(l) + ': ' + str(self.b[l].shape[1]) + ' |')
            print(36*'-')

    def __feedforward(self, X):
        A, H = [], [X]
        for l in range(len(self)):
            A.append(H[-1] @ self.W[l] + self.b[l])
            H.append(self.act[l](A[-1]))
        return A, H  # pre-activation, post-activation (i.e., hypothesis at each layer)

    def __backprop(self, y, A, H):
        dW, db = [], []

        loss = np.mean((H[-1] - y) ** 2)

        delta = [-np.array((y-H[-1]) * activations.relu_prime(A[-1])).squeeze()]
        dW.append(H[-2].T * delta[-1])
        db.append(delta[-1])

        for l in range(len(self)-1, 0, -1):
            delta.append(-np.dot(delta[-1], self.W[l].T) * activations.relu_prime(A[-l-1]))
            dW.insert(0, H[-l-2].T * delta[-1])
            db.insert(0, delta[-1])

        return loss, dW, db

    def fit(self, X, y, settings={'loss': 'mse', 'optimizer': 'sgd', 'lr': 0.01, 'epochs': 100}):
        for epoch in range(settings['epochs']):
            for (data_X, data_y) in data_loader(X, y):
                ## Forward pass
                A, H = self.__feedforward(data_X)
                ## Backward pass
                loss, dW, db = self.__backprop(data_y, A, H)
                print('loss at epoch {}: {}'.format(epoch, loss))
                ## Update weights
                for l in range(len(self)):
                    self.W[l] -= settings['lr'] * dW[l]
                    self.b[l] -= settings['lr'] * db[l]
    def predict(self, X):
        A, H = self.__feedforward(X)
        return H[-1]
