import numpy as np

### TANH: x -> (e^x - e^-x) / (e^x + e^-x)
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_prime(x):
    return 1 - tanh(x) ** 2

### SIGMOID: x -> 1 / (1 + e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

### SOFTMAX: x -> exp(x) / sum(exp(x))
def softmax(x):
    exps = [np.exp(i) for i in x]
    sum_exps = sum(exps)
    return [i / sum_exps for i in exps]

def softmax_prime(x):
    pass

### LINEAR: x -> x
def linear(x):
    return x

def linear_prime(x):
    return 1

### RELU: x -> max(0, x)
def relu(x):
    return x if x > 0 else 0

def relu_prime(x):
    return 1 if x > 0 else 0

### LEAKY RELU: x -> max(alpha * x, x)
def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_prime(x, alpha=0.01):
    return 1 if x > 0 else alpha
