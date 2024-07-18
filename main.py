import numpy as np

from utils.preprocessing import train_test_split
from utils.activations import relu
from utils.mlp import MLP


## Synthetic data
n_samples, n_features = 1024, 1
X = .7 * np.random.randn(n_samples, n_features) + 2.
y = -1.5 * np.sum(X, axis=1) + 5.  # alpha, beta = -1.5, 5.0
(X_train, X_test), (y_train, y_test) = train_test_split(X, y)
model = MLP([[n_features, 1, relu]])
model.summary()
settings = {'lr': 0.01, 'epochs': 16}
model.fit(X_train, y_train, settings=settings)
y_hat = model.predict(X_test)  # predicted output
## MSE
mse = np.mean((y_test - y_hat) ** 2)
print(mse)
print('alpha = {} and beta = {}'.format(model.W[0].squeeze(), model.b[0].squeeze()))