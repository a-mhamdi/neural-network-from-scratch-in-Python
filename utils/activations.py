import numpy as np


### TANH: x -> (e^x - e^-x) / (e^x + e^-x)
def tanh(x, diff=False):
    if diff:
        return 1 - tanh(x) ** 2
    else:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


### SIGMOID: x -> 1 / (1 + e^-x)
def sigmoid(x, diff=False):
    if diff:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return 1 / (1 + np.exp(-x))


### SOFTMAX: x -> exp(x) / sum(exp(x))
def softmax(x, diff=False):
    if diff:
        return softmax(x)/(1-softmax(x))
    else:
        exps = [np.exp(i) for i in x]
        sum_exps = sum(exps)
        return [i / sum_exps for i in exps]


### LINEAR: x -> x
def linear(x, diff=False):
    if diff:
        return np.ones(x.shape)
    else:
        return x


### RELU: x -> max(0, x)
def relu(x, diff=False):
    if diff:
        return np.where(x >= 0, 1, 0)
    else:
        return np.maximum(x, 0)


### LEAKY RELU: x -> max(alpha * x, x)
def leaky_relu(x, alpha=0.01, diff=False):
    if diff:
        return np.where(x >= 0, 1, alpha)
    else:
        return np.maximum(alpha * x, x)

