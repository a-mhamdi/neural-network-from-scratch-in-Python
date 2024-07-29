import numpy as np

from utils.preprocessing import data_loader
from utils import activations

default_hparams = {'loss': 'mse', 'optimizer': 'sgd', 'lr': 0.01, 'regularizer': 'None', 'lambda': 0.01, 'r': 0.5, 'dropout': 0.}


def loss_fct(y, yhat, hparams=default_hparams):
    if hparams['loss'].lower() == 'mae':
        loss = np.mean(np.abs(y - yhat))
    elif hparams['loss'].lower() == 'mse':
        loss = np.mean((y - yhat) ** 2)
    elif hparams['loss'].lower() == 'rmse':
        loss = np.sqrt(np.mean((y - yhat) ** 2))
    elif hparams['loss'].lower() == 'cross-entropy':
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))

    return loss


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

    def __feedforward(self, X, hparams=default_hparams):
        A, H = [], [X]
        for l in range(len(self)):
            A.append(H[-1] @ self.W[l] + self.b[l])
            H.append(self.act[l](A[-1]))
        return A, H  # pre-activation, post-activation (i.e., hypothesis at each layer)

    def __backprop(self, y, A, H, hparams=default_hparams):
        loss, dW, db = 0., [], []

        delta = [np.array((y-H[-1]) * self.act[-1](A[-1], diff=True)).squeeze()]
        dW.append(-H[-2].T * delta[-1])
        db.append(-delta[-1])

        for l in range(len(self)-1, -1, -1):
            delta.append(self.act[l](A[l], diff=True) * np.dot(self.W[l+1].T, delta[-1]))
            dW.insert(0, -H[-l-2].T * delta[-1])
            db.insert(0, -delta[-1])

        loss = loss_fct(y, H[-1], hparams=hparams)
        for l in range(len(self)):
            if hparams['regularizer'].lower() == 'l1':
                loss += hparams['lambda'] * np.sum(np.abs(self.W[l]))
                dW[l] += hparams['lambda'] * np.sign(self.W[l])
            elif hparams['regularizer'].lower() == 'l2':
                loss += 0.5 * hparams['lambda'] * np.sum(self.W[l] ** 2)
                dW[l] += hparams['lambda'] * self.W[l]
            elif hparams['regularizer'].lower() == 'elasticnet':
                loss += hparams['lambda'] * hparams['r'] * np.sum(np.abs(self.W[l])) + hparams['lambda'] * (1-hparams['r']) * np.sum(self.W[l] ** 2)
                dW[l] += hparams['lambda'] * hparams['r'] * np.sign(self.W[l]) + hparams['lambda'] * (1-hparams['r']) * self.W[l]

        return loss, dW, db

    def fit(self, X, y, hparams=default_hparams):
        for (data_X, data_y) in data_loader(X, y):
            ## Forward pass
            A, H = self.__feedforward(data_X, hparams=hparams)
            ## Backward pass
            loss, dW, db = self.__backprop(data_y, A, H, hparams=hparams)
            print('loss >>> train {}'.format(loss))
            ## Update weights
            for l in range(len(self)):
                self.W[l] -= hparams['lr'] * dW[l]
                self.b[l] -= hparams['lr'] * db[l]

    def predict(self, X):
        A, H = self.__feedforward(X)

        return H[-1]