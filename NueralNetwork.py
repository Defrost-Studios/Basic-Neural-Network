import numpy as np
import pandas as pd
import struct
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


data = pd.read_csv('mnist_full.csv')  # Loads mnist dataset here

data = np.array(data)
m, n = data.shape
np.random.seed(42)
np.random.shuffle(data)  # Shuffles the dataset

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0

def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((128, 1))
    W2 = np.random.randn(64, 128) * np.sqrt(2 / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b3 = np.zeros((10, 1))
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def deriv_ReLU(Z):
    return Z > 0
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha *dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 25 == 0:
            print("iteration", i)
            print("accuracy", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3

alpha = 0.1
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 1000, alpha)

_, _, _, _, _, A3_train = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
_, _, _, _, _, A3_dev = forward_prop(W1, b1, W2, b2, W3, b3, X_dev)

print("final training accuracy", get_accuracy(get_predictions(A3_train), Y_train))
print("validation accuracy", get_accuracy(get_predictions(A3_dev), Y_dev))

# Returns the test results using the test data we originally split up from the total dataset
'''
_, _, _, _, _, A3_test = forward_prop(W1, b1, W2, b2, W3, b3, X_dev)
print("test accuracy", get_accuracy(get_predictions(A3_test), Y_dev)) 
'''
# Returns a Confusion matrix Table
'''
cm = confusion_matrix(Y_dev, get_predictions(A3_dev), labels=np.arange(10))
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(label="Number of images")
plt.xticks(np.arange(10), np.arange(10))
plt.yticks(np.arange(10), np.arange(10))
threshold = cm.max() / 2
for row in range(10):
    for column in range(10):
        plt.text(
            column,
            row,
            cm[row, column],
            ha="center",
            va="center",
            color="white" if cm[row, column] > threshold else "black",
        )
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("MNIST Validation Confusion Matrix")
plt.tight_layout()
plt.show()
'''