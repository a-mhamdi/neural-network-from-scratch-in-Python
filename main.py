import numpy as np
from matplotlib import pyplot as plt

from utils.preprocessing import train_test_split
from utils.activations import relu
from utils.mlp import MLP, loss_fct


## Synthetic data
n_samples, n_features, epochs = 1024, 3, 8
X = .7 * np.random.rand(n_samples, n_features) + 2.
y = -1.5 * np.sum(X, axis=1) + 5.
(X_train, X_test), (y_train, y_test) = train_test_split(X, y)
model = MLP([[n_features, 4, relu], [4, 7, relu], [7, 1, relu]])
model.summary()
settings = {'loss': 'rmse', 'optimizer': 'sgd', 'lr': 0.01, 'regularizer': 'None', 'lambda': 0.01, 'r': 0.5, 'dropout': 0.}

ltrn, ltst = [], []
for epoch in range(epochs):
    model.fit(X_train, y_train, hparams=settings)
    ytrn_hat = model.predict(X_train)  # predicted `train` output
    ltrn.append(loss_fct(y_train, ytrn_hat, hparams=settings))
    ytst_hat = model.predict(X_test)  # predicted `test` output
    ltst.append(loss_fct(y_test, ytst_hat, hparams=settings))

plt.plot(ltrn, label='train')
plt.plot(ltst, label='test')
plt.legend()
plt.show()

