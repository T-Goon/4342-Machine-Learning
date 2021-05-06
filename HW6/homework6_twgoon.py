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
    return wp1, bp1, wp2, bp2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    vec = np.concatenate((W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()))
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
    return ones.T

def predict(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = np.dot(W1, X).T + b1
    z1 = z1.T
    h1 = np.zeros(z1.shape)
    pos = z1 > 0
    h1[pos] = z1[pos]

    z2 = np.dot(W2, h1).T + b2
    z2 = z2.T
    yh = np.exp(z2-np.amax(z2)) / np.sum(np.exp(z2-np.amax(z2)), axis=0)

    return yh, z1, h1

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):
    yh, z1, h1 = predict(X, w)

    return -np.sum(Y * np.log(yh)) / Y.shape[1]

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w, l1, l2):
    W1, b1, W2, b2 = unpack(w)

    yh, z1, h1 = predict(X, w)

    relu_p = np.zeros(z1.T.shape)
    pos = z1.T > 0
    relu_p[pos] = 1
    
    g_T = np.dot((yh - Y).T, W2) * relu_p

    return pack(
    (np.dot(g_T.T, X.T) / X.shape[1]) + (l1 * np.sign(W1)) + (l2 * W1), 
    # (np.dot(g_T.T, X.T) / X.shape[1]) + (L2reg * W1) + (L1reg * np.sign(W1)),
    np.mean(g_T.T, axis=1), 
    (np.dot((yh - Y), h1.T) / X.shape[1]) + (l1 * np.sign(W2)) + (l2 * W2), 
    # (np.dot((yh - Y), h1.T) / X.shape[1]) + (L2reg * W2) + (L1reg * np.sign(W2)), 
    np.mean((yh - Y), axis=1))

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN.
def train (trainX, trainY, testX, testY, w, lr, batch_size, num_epochs, l1, l2):
    num_batches = trainX.shape[1]//batch_size

    for _ in range(num_epochs):
        for j in range(num_batches-1):
            g = gradCE(trainX[:, j*batch_size : (j+1)*batch_size], 
            trainY[:, j*batch_size : (j+1)*batch_size], 
            w, l1, l2)
            w = w - (lr * g)
            
    return w

# caculate the accuracy of the model
def calc_accuracy(testX, testY, w):
    predictions = predict(testX, w)
    predictions = np.argmax(predictions[0], axis=0)
    labels = np.argmax(testY, axis=0)

    return np.mean(predictions == labels)

# loop through different hyperparameter values to find the best one
def findBestHyperparameters(validX, validY):
    lr = [.001, .005, .01]
    batch_size = [32, 64, 128]
    num_epochs = [10, 20, 30]
    L1reg = [0, .01, .05]
    L2reg = [0, .01, .05]

    acc_best = 0
    lr_best = None
    batch_size_best = None
    num_epochs_best = None
    L1reg_best = None
    L2reg_best = None

    for e in lr:
        for bs in batch_size:
            for ne in num_epochs:
                for l1 in L1reg:
                    for l2 in L2reg:

                        # Initialize weights randomly
                        W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
                        b1 = 0.01 * np.ones(NUM_HIDDEN)
                        W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
                        b2 = 0.01 * np.ones(NUM_OUTPUT)

                        # Concatenate all the weights and biases into one vector; this is necessary for check_grad
                        w = pack(W1, b1, W2, b2)

                        w = train(validX, validY, None, None, w, e, bs, ne, l1, l2)

                        acc = calc_accuracy(validX, validY, w)
                        print("Learning Rate: ", e)
                        print("Batch Size: ", bs)
                        print("Num Epochs: ", ne)
                        print("l1 strength: ", l1)
                        print("l2 strength: ", l2)
                        print("Accuracy: ", acc)
                        print()
                        if(acc > acc_best):
                            acc_best = acc
                            lr_best = e
                            batch_size_best = bs
                            num_epochs_best = ne
                            L1reg_best = l1
                            L2reg_best = l2

    print("Learning Rate Best: ", lr_best)
    print("Batch Size Best: ", batch_size_best)
    print("Num Epochs Best: ", num_epochs_best)
    print("l1 strength Best: ", L1reg_best)
    print("l2 strength Best: ", L2reg_best)
    print("Accuracy Best: ", acc_best)
    print()
    return lr_best, batch_size_best, num_epochs_best, L1reg_best, L2reg_best

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        trainX = trainX.T
        testX = testX.T
        trainY = one_hot(trainY, 10)
        testY = one_hot(testY, 10)

        # squash values
        trainX = trainX / 255
        testX = testX / 255

        # shuffle the data
        trainData = np.concatenate((trainX, trainY)).T
        testData = np.concatenate((testX, testY)).T

        np.random.shuffle(trainData)
        np.random.shuffle(testData)

        trainX = trainData.T[:-10, :]
        trainY = trainData.T[-10:, :]
        testX = testData.T[:-10, :]
        testY = testData.T[-10:, :]

        # create a validation set
        validX = trainX[:, int(trainX.shape[1]*.8):]
        validY = trainY[:, int(trainY.shape[1]*.8):]
        trainX = trainX[:, :int(trainX.shape[1]*.8)]
        trainY = trainY[:, :int(trainY.shape[1]*.8)]
        
    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]

    w1p, b1p, w2p, b2p = unpack(scipy.optimize.approx_fprime(w, lambda W_: fCE(trainX[:, idxs], trainY[:, idxs], W_), 1e-8))
    w1p2, b1p2, w2p2, b2p2 = unpack(gradCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w, 0, 0))
    print(np.sum(w1p - w1p2))
    print()
    print(np.sum(b1p - b1p2))
    print()
    print(np.sum(w2p - w2p2))
    print()
    print(np.sum(b2p - b2p2))
    print()

    print("check grad: ", scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[:, idxs]), np.atleast_2d(trainY[:, idxs]), w_, 0, 0), \
                                    w))

    lr_best, batch_size_best, num_epochs_best, L1reg_best, L2reg_best = \
    findBestHyperparameters(validX, validY)

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    # Concatenate all the weights and biases into one vector; this is necessary for check_grad
    w = pack(W1, b1, W2, b2)

    # Train the network using SGD.
    w = train(trainX, trainY, testX, testY, w, lr_best, batch_size_best, num_epochs_best, L1reg_best, L2reg_best)
    # w = train(trainX, trainY, testX, testY, w, .01, 64, 10, .01, .01)

    # calculate the model's accuracy
    print("CE Loss: ", fCE(testX, testY, w))
    print("Accuracy: ", calc_accuracy(testX, testY, w))

