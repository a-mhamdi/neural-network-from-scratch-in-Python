from sklearn import datasets

import numpy as np
from matplotlib import pyplot as plt

from utils.preprocessing import train_test_split
from utils.mlp import MLP
from utils.activations import relu, sigmoid
from utils.metrics import loss_fct, accuracy

np.random.seed(1234)

# Dataset Load
ds = datasets.load_breast_cancer()
X, y = ds.data, ds.target

n_features, epochs = X.shape[1], 16

try:
    n_targets = y.shape[1]
except IndexError:
    n_targets = 1

(X_train, X_test), (y_train, y_test) = train_test_split(X, y)

# Standardization/ Normalization
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
# X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

mn, mx = np.min(X_train, axis=0), np.max(X_train, axis=0)
X_train, X_test = (X_train - mn) / (mx - mn), (X_test - mn) / (mx - mn)

# Model Design
model = MLP([[n_features, 16, relu], [8, relu], [4, relu], [n_targets, sigmoid]])
model.summary()

settings = {
    'batch_size': 12,
    'loss': 'bce',
    'optimizer': 'sgd',
    'lr': 1e-2,
    'regularization': 'none',
    'lambda': 0.1,
    'r': 0.5
}

ltrn, ltst = [], []
for epoch in range(epochs):
    model.fit(X_train, y_train, hparams=settings)
    ytrn_hat = model.predict(X_train)  # predicted `train` output
    ltrn.append(loss_fct(y_train, ytrn_hat, hparams=settings))
    ytst_hat = model.predict(X_test)  # predicted `test` output
    ltst.append(loss_fct(y_test, ytst_hat, hparams=settings))

    print(f"{accuracy(y_train, ytrn_hat): .3f}", f"{accuracy(y_test, ytst_hat): .3f}", sep=' | ')

fig, ax = plt.subplot_mosaic([['a', 'b'], ['a', 'c']], layout='constrained')

ax['a'].set_title('loss')
ax['a'].plot(ltrn, label='train')
ax['a'].plot(ltst, label='test')
ax['a'].legend()

ax['b'].set_title('actual vs. predicted (train)')
ax['b'].scatter(range(len(y_train)), y_train)
ax['b'].scatter(range(len(ytrn_hat)), (ytrn_hat >= .5) * 1)

ax['c'].set_title('actual vs. predicted (test)')
ax['c'].scatter(range(len(y_test)), y_test)
ax['c'].scatter(range(len(ytst_hat)), (ytst_hat >= .5) * 1)

plt.show()
