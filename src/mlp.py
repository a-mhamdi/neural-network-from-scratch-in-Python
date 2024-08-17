import numpy as np

from src.preprocessing import data_loader
from src.activations import relu, leaky_relu, tanh, sigmoid, softmax
from src.metrics import loss_fct


class MLP:
    def __init__(self, arch: list) -> None:
        leng = len(arch)
        self.leng = leng

        self.w = [None] * leng
        self.b = [None] * leng
        self.act = [None] * leng

        self.__init_weights(arch)

        self.cache = {'z': [], 'a': []}

        self.gradients = {
            'w': [np.zeros(w.shape) for w in self.w],
            'b': [np.zeros(b.shape) for b in self.b]
        }

    def __init_weights(self, arch: list) -> None:
        for layer in range(self.leng):
            if layer == 0:
                self.act[layer] = arch[layer][2]
                size_out, size_in = arch[layer][0], arch[layer][1]
                self.b[layer] = np.zeros(size_in)
            else:
                self.act[layer] = arch[layer][1]
                size_out, size_in = self.b[layer - 1].shape[0], arch[layer][0]
                self.b[layer] = np.zeros(size_in)

            if self.act[layer] in [tanh, sigmoid, softmax]:
                # `Glorot/Xavier` Normal
                sigma = np.sqrt(2 / float(size_out + size_in))
                self.w[layer] = np.random.normal(0, sigma, size=(size_in, size_out))
            elif self.act[layer] in [relu, leaky_relu]:
                # `He` Normal
                sigma = np.sqrt(2 / float(size_out))
                self.w[layer] = np.random.normal(0, sigma, size=(size_in, size_out))
            else:
                # `LeCun` Normal
                sigma = np.sqrt(1 / float(size_out))
                self.w[layer] = np.random.normal(0, sigma, size=(size_in, size_out))

    def __len__(self) -> int:
        return self.leng

    def summary(self) -> None:
        tot_params = 0
        print(70*'=', 30*' ' + 'Model Summary', sep='\n')
        print(70*'-')
        for layer in range(len(self)):
            msg = '| Layer #' + str(layer) + ' => ' \
                  + 'W' + str(layer) + ': ' + str(self.w[layer].shape) \
                  + ' and b' + str(layer) + ': ' + str(self.b[layer].shape[0])

            params = np.prod(self.w[layer].shape) + self.b[layer].shape[0]
            tot_params += params
            str_params = str(params) + ' trainable parameters'
            print(msg + (40-len(msg))*' ' + ' | ' + str_params + (25-len(str_params))*' ' + ' |')
            print(70*'-')
        print('Total trainable parameters: ' + str(tot_params), 70*'=', sep='\n')

    def __feedforward(self, x: np.ndarray) -> None:
        z, a = [], [x]
        for layer in range(len(self)):
            z.append(self.w[layer] @ a[-1] + self.b[layer])
            a.append(self.act[layer](z[-1]))

        # Store cache (useful for backprop)
        self.cache['z'] = z  # pre-activation
        self.cache['a'] = a  # post-activation (i.e., at each layer: a ≜ f(z), where z ≜ w @ a + b)

    def __backprop(self, y: np.ndarray, hparams: dict) -> float:
        # Unpack the cache
        a, z = self.cache['a'], self.cache['z']

        leng = len(self)
        dw, db = [np.zeros(w.shape) for w in self.w], [np.zeros(b.shape) for b in self.b]
        delta = [np.zeros(b.shape) for b in self.b]

        loss, dloss = loss_fct(y, a[-1], hparams, is_dloss=True)

        for layer in range(leng-1, -1, -1):
            if layer == leng-1:
                delta[layer] = dloss * self.act[layer](z[layer], diff=True)
            else:
                # δ ≜ (δ @ w) * f'(z)
                delta[layer] = (delta[layer+1] @ self.w[layer+1]) * self.act[layer](z[layer], diff=True)
            # Gradients of J wrt w and b: 𝜕J/𝜕w, 𝜕J/𝜕b
            dw[layer] = -np.reshape(delta[layer], (-1, 1)) * np.reshape(a[layer], (1, -1))
            db[layer] = -delta[layer]

        if hparams['regularization'].lower() == 'l1':
            for layer in range(leng):
                loss += hparams['lambda'] * np.sum(np.abs(self.w[layer]))
                dw[layer] += hparams['lambda'] * np.sign(self.w[layer])
        elif hparams['regularization'].lower() == 'l2':
            for layer in range(leng):
                loss += 0.5 * hparams['lambda'] * np.sum(self.w[layer] ** 2)
                dw[layer] += hparams['lambda'] * self.w[layer]
        elif hparams['regularization'].lower() == 'elasticnet':
            for layer in range(leng):
                loss += hparams['lambda'] * (hparams['r'] * np.sum(np.abs(self.w[layer]))
                                             + (1-hparams['r']) * np.sum(self.w[layer] ** 2))
                dw[layer] += hparams['lambda'] * (hparams['r'] * np.sign(self.w[layer])
                                                  + (1-hparams['r']) * self.w[layer])

        # Store gradients
        for layer in range(leng):
            self.gradients['w'][layer] += dw[layer] / hparams['batch_size']  # 𝜕J/𝜕w
            self.gradients['b'][layer] += db[layer] / hparams['batch_size']  # 𝜕J/𝜕b

        return loss

    def fit(self, x: np.ndarray, y: np.ndarray, hparams: dict) -> None:
        batch_size = hparams['batch_size']
        for (data_x, data_y) in data_loader(x, y, batch_size=batch_size):
            cost = 0
            for i in range(data_x.shape[0]):
                # Forward pass
                self.__feedforward(data_x[i])
                # Backward pass
                cost += self.__backprop(data_y[i], hparams=hparams)

            cost /= batch_size
            # print('loss >>> train {}'.format(cost))
            # Update weights
            for layer in range(len(self)):
                self.w[layer] -= hparams['lr'] * self.gradients['w'][layer]
                self.b[layer] -= hparams['lr'] * self.gradients['b'][layer]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        leng, n_targets = x.shape[0], self.b[-1].shape[0]
        out = np.zeros((leng, n_targets))
        for i in range(leng):
            data = x[i]
            self.__feedforward(data)
            out[i] = self.cache['a'][-1]

        return out
