from sklearn import datasets

import numpy as np
from matplotlib import pyplot as plt

from src.preprocessing import train_test_split
from src.mlp import MLP
from src.activations import relu, sigmoid, linear
from src.metrics import loss_fct, r2_score, accuracy, precision, recall, f1_score, cm

np.random.seed(1234)

# Dataset Load
ds = datasets.load_breast_cancer()
X, y = ds.data, ds.target.reshape(-1, 1)

n_features, epochs = X.shape[1], 16

try:
    n_targets = y.shape[1]
except IndexError:
    n_targets = 1

# Data Split
(X_train, X_test), (y_train, y_test) = train_test_split(X, y)

# Standardization/ Normalization
mu, std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
# X_train, X_test = (X_train - mu) / std, (X_test - mu) / std

mn, mx = np.min(X_train, axis=0), np.max(X_train, axis=0)
X_train, X_test = (X_train - mn) / (mx - mn), (X_test - mn) / (mx - mn)

# Model Design
model = MLP([[n_features, 20, relu], [20, relu], [20, relu], [5, relu], [n_targets, sigmoid]])
model.summary()

settings = {
    'batch_size': 8,
    'loss': 'mse',
    'optimizer': 'sgd',
    'lr': 1e-3,
    'regularization': 'none',
    'lambda': 0.3,
    'r': 0.5
}

ltrn, ltst = [], []
for epoch in range(epochs):
    model.fit(X_train, y_train, hparams=settings)
    ytrn_hat = model(X_train)  # predicting the `train` set
    ltrn.append(loss_fct(y_train, ytrn_hat, hparams=settings))
    ytst_hat = model(X_test)  # predicting the `test` set
    ltst.append(loss_fct(y_test, ytst_hat, hparams=settings))

print(f"Accuracy: {accuracy(y_train, ytrn_hat): .3f}", f"{accuracy(y_test, ytst_hat): .3f}", sep=' | ')
print(f"Precision: {precision(y_train, ytrn_hat): .3f}", f"{precision(y_test, ytst_hat): .3f}", sep=' | ')
print(f"Recall: {recall(y_train, ytrn_hat): .3f}", f"{recall(y_test, ytst_hat): .3f}", sep=' | ')
print(f"F1-score: {f1_score(y_train, ytrn_hat): .3f}", f"{f1_score(y_test, ytst_hat): .3f}", sep=' | ')
print("Confusion Matrix", 16*"=", "TRAIN SET",  cm(y_train, ytrn_hat), "TEST SET", cm(y_test, ytst_hat), sep='\n')

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
