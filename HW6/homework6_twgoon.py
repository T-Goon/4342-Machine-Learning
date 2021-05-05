import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import percentile
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    wp1 = w[:31360].reshape((40, 784))
    bp1 = w[31360:31400]
    wp2 = w[31400:31800].reshape((10, 40))
    bp2 = w[31800:]
    print()
    print(wp1.shape)
    print(bp1.shape)
    print(wp2.shape)
    print(bp2.shape)
    print()
    return wp1, bp1, wp2, bp2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    vec = np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))
    # print("vec shape: ",vec.shape)
    return vec

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("data/{}/fashion_mnist_{}_images.npy".format(which, which))
    labels = np.load("data/{}/fashion_mnist_{}_labels.npy".format(which, which))
    return images, labels

# convert 1d array to array of one-hot vectors
def one_hot(y, c):
    # create the array
    ones = np.zeros((y.shape[0], c))

    # make array one hot
    ones[np.arange(y.shape[0]), y.astype(int)] = 1
    return ones

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = np.dot(W1, X) + b1
    h1 = np.zeros(40)
    pos = z1 > 0
    h1[pos] = z1[pos]

    z2 = np.dot(W2, h1) + b2
    yh = np.exp(z2 - np.amax(z2)) / np.sum(np.exp(z2-np.amax(z2)))
    
    y_1h = one_hot(Y, 10)

    return -np.sum(yh * np.log(Y)) / Y.shape[0]

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    W1, b1, W2, b2 = unpack(w)
    return grad

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w):
    pass

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")

    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    print()
    print(W1.shape)
    print(b1.shape)
    print(W2.shape)
    print(b2.shape)
    print()

    print()
    print(W1[0, 0])
    print(b1[0])
    print(W2[0, 0])
    print(b2[0])
    print()
    
    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)
    w = unpack(w)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    # idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    # print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]), np.atleast_2d(trainY[idxs,:]), w_), \
    #                                 w))

    # # Train the network using SGD.
    # train(trainX, trainY, testX, testY, w)
