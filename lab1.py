# ANN Lab 1
# Perceptron for Spam Email Detection

import numpy as np

# Step Activation Function
def step(x):
    if x >= 0:
        return 1
    return 0

# Training Data
# Features:
# X1 = Contains Link
# X2 = Contains FREE keyword
# X3 = Has Attachment

X = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 0]
])

y = np.array([0, 1, 0, 1, 1, 0, 1, 0])

# Initialize Parameters
weights = np.zeros(3)
bias = 0
learning_rate = 0.1
epochs = 20

# Training
for epoch in range(epochs):
    for i in range(len(X)):
        x = X[i]
        target = y[i]

        net = np.dot(x, weights) + bias
        output = step(net)

        error = target - output

        weights = weights + learning_rate * error * x
        bias = bias + learning_rate * error

# Testing
print('Final Weights:', weights)
print('Final Bias:', bias)

print('\nPredictions:')
for x in X:
    net = np.dot(x, weights) + bias
    prediction = step(net)
    print(x, '->', prediction)
