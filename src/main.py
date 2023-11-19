import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import matplotlib.pyplot as plt
import time
import pandas as pd

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = .1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # def backward(self, dvalues):
    #     self.dweights = np.dot(self.inputs.T, dvalues)
    #     self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
    #
    #     self.dinputs = np.dot(dvalues, self.weights.T)


class Actiavtion_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    # def backward(self, dvalues):
    #     drelu = np.where(self.inputs > 0, 1, 0)
    #     self.dinputs = dvalues * drelu


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        self.inputs = inputs

    # def backward(self, dvalues):
    #     batch_size = len(self.output)
    #
    #     dsoftmax = np.zeros_like(self.inputs)
    #
    #     for i, (softmax, dvalue) in enumerate(zip(self.output, dvalues)):
    #         softmax = softmax.reshape(-1, 1)
    #         jacobian_matrix = np.diagflat(softmax) - np.dot(softmax, softmax.T)
    #         dsoftmax[i, :] = np.dot(jacobian_matrix, dvalue)
    #
    #     self.dinputs = dsoftmax / batch_size


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CatagoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # def backward(self, output, y):
    #     gradients = self.gradient(output, y)
    #     return gradients

    # def gradient(self, output, y):
    #     samples = len(output)
    #
    #     if len(y.shape) == 1:
    #         y_true = np.eye(len(output[0]))[y]
    #     else:
    #         y_true = y
    #
    #     grad = (output -y_true) / samples
    #
    #     return grad


# Data
#X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)

# init neurons
dense1 = Layer_Dense(2, 3)
activation1 = Actiavtion_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

loss_function = Loss_CatagoricalCrossentropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.weights.copy()

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')

# # # optimizer
# dense1.weights += 0.05 * np.random.randn(2, 3)
# dense1.biases += 0.05 * np.random.randn(1, 3)
# dense2.weights += 0.05 * np.random.randn(3, 3)
# dense2.biases += 0.05 * np.random.randn(1, 3)

learning_rate = 0.1

for iteration in range(1000000):

    # NEED TO HAVE OPTIMIZER BEFORE CALCULATIONS
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)

    accuracy = np.mean(predictions == y)

    # Backward pass
    #activation2.backward(loss_function.gradient(activation2.output, y))
    #dense2.backward(activation2.dinputs)
    #activation1.backward(dense2.inputs)
    #dense1.backward(activation1.dinputs)

    learning_rate *= 0.99

    # # Optimizer
    # dense1.weights -= learning_rate * dense1.dweights
    # dense1.biases -= learning_rate * dense1.dbiases
    # dense2.weights -= learning_rate * dense2.dweights
    # dense2.biases -= learning_rate * dense2.dbiases



    if loss < lowest_loss:
        print('New set of weights found iteration:', iteration, 'loss', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

        # writes the best data
        df = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'class': predictions})
        df.to_csv('data.csv', index=False)

    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

        # df = pd.DataFrame({'X': X[:, 0], 'Y': X[:, 1], 'class': predictions})
        # df.to_csv('data.csv', index=False)

plt.scatter(X[:, 0], X[:, 1], c=predictions, s=40, cmap='brg')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='brg')
plt.show()
