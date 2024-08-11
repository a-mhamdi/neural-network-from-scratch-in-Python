import numpy as np

from utils.preprocessing import data_loader
from utils.activations import relu, leaky_relu, sigmoid, softmax
from utils.metrics import loss_fct


class MLP():
    def __init__(self, arch):
        leng = len(arch)
        self.W = [None] * leng
        self.b = [None] * leng
        self.act = [None] * leng
        self.leng = leng
        for l in range(leng):
            if l == 0:
                self.act[l] = arch[l][2]
                size_out, size_in = arch[l][0], arch[l][1]
                self.b[l] = np.zeros(arch[l][1])
            else:
                self.act[l] = arch[l][1]
                size_out, size_in = self.b[l - 1].shape[0], arch[l][0]
                self.b[l] = np.zeros(arch[l][0])

            if self.act[l] == sigmoid or self.act[l] == softmax:
                # `Glorot/Xavier` Normal
                limit = np.sqrt(2 / float(size_out + size_in))
                self.W[l] = np.random.normal(0., limit, size=(size_in, size_out))
            elif self.act[l] == relu or self.act[l] == leaky_relu:
                # `He` Normal
                limit = np.sqrt(2 / float(size_out))
                self.W[l] = np.random.normal(0., limit, size=(size_in, size_out))
            else:
                # LeCun Normal
                limit = np.sqrt(1 / float(size_out))
                self.W[l] = np.random.normal(0., limit, size=(size_in, size_out))

    def __len__(self):
        return self.leng

    def summary(self):
        tot_params = 0
        print(70*'=', 30*' ' + 'Model Summary', sep='\n')
        print(70*'-')
        for l in range(len(self)):
            msg = '| Layer #' + str(l) + ' => ' \
                  + 'W' + str(l) + ': ' + str(self.W[l].shape) \
                  + ' and b' + str(l) + ': ' + str(self.b[l].shape[0])

            params = np.prod(self.W[l].shape) + self.b[l].shape[0]
            tot_params += params
            str_params = str(params) + ' trainable parameters'
            print(msg + (40-len(msg))*' ' + ' | ' + str_params + (25-len(str_params))*' ' + ' |')
            print(70*'-')
        print('Total trainable parameters: ' + str(tot_params), 70*'=', sep='\n')

    def __feedforward(self, x):
        z, a = [], [x.squeeze()]
        for l in range(len(self)):
            z.append((self.W[l] @ a[-1]) + self.b[l])
            a.append(self.act[l](z[-1]))

        return z, a  # pre-activation, post-activation (i.e., at each layer: a = f(z), where z = W @ a + b)

    def __backprop(self, z, a, y, hparams):
        dW, db = [], []
        loss, dlt = loss_fct(y, a[-1], hparams, is_dlt=True)
        delta = [np.array(dlt * self.act[-1](z[-1], diff=True))]
        for l in range(len(self)-2, -1, -1):
            delta.append((delta[-1] @ self.W[l+1]) * self.act[l](z[l], diff=True))
        delta = delta[::-1]

        for l in range(len(self)):
            dW.append(-np.reshape(delta[l], (-1, 1)) * np.reshape(a[l], (1, -1)))
            db.append(-delta[l])

        if hparams['regularization'].lower() == 'l1':
            for l in range(len(self)):
                loss += hparams['lambda'] * np.sum(np.abs(self.W[l]))
                dW[l] += hparams['lambda'] * np.sign(self.W[l])
        elif hparams['regularization'].lower() == 'l2':
            for l in range(len(self)):
                loss += 0.5 * hparams['lambda'] * np.sum(self.W[l] ** 2)
                dW[l] += hparams['lambda'] * self.W[l]
        elif hparams['regularization'].lower() == 'elasticnet':
            for l in range(len(self)):
                loss += hparams['lambda'] * (hparams['r'] * np.sum(np.abs(self.W[l])) + (1-hparams['r']) * np.sum(self.W[l] ** 2))
                dW[l] += hparams['lambda'] * (hparams['r'] * np.sign(self.W[l]) + (1-hparams['r']) * self.W[l])

        return loss, dW, db

    def fit(self, x, y, hparams):
        batch_size = hparams['batch_size']
        for (data_x, data_y) in data_loader(x, y, batch_size=batch_size):
            loss, dW, db = 0., [np.zeros(w.shape) for w in self.W], [np.zeros(b.shape) for b in self.b]
            for i in range(data_x.shape[0]):
                ## Forward pass
                z, a = self.__feedforward(data_x[i])
                ## Backward pass
                l, gradW, gradb = self.__backprop(z, a, data_y[i], hparams=hparams)
                loss += l
                for l in range(len(self)):
                    dW[l] += gradW[l]
                    db[l] += gradb[l]

            loss /= batch_size
            print('loss >>> train {}'.format(loss))
            ## Update weights
            for l in range(len(self)):
                self.W[l] -= hparams['lr'] * dW[l] / batch_size
                self.b[l] -= hparams['lr'] * db[l] / batch_size

    def predict(self, x):
        leng, n_targets = x.shape[0], self.b[-1].shape[0]
        out = np.zeros((leng, n_targets))
        for i in range(leng):
            data = x[i, :]
            _, a = self.__feedforward(data)
            out[i, :] = a[-1]

        return out
