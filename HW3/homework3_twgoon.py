import numpy as np
import matplotlib.pyplot as plt
import sys

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def append1s (imgs):

    # Append one onto each image
    ones = np.ones((imgs.shape[0], 1))
    imgs_b = np.concatenate((imgs, ones), axis=1)

    imgs_b = imgs_b.T

    return imgs_b

# convert 1d array to array of one-hot vectors
def one_hot(y, c):
    # create the array
    ones = np.zeros((y.shape[0], c))

    # make array one hot
    ones[np.arange(y.shape[0]), y.astype(int)] = 1
    return ones

def CE_loss(w, imgs, y, classes):
    z = np.dot(imgs.T, w)

    # convert to probabilities
    p = np.exp(z) / np.sum(np.exp(z))
    y_h1 = one_hot(y, classes)

    # removed 1s from the vector
    p = np.where(p != 0, p, 1**-10)

    # print("p shape",  p.shape)
    # print("y shape", y_h1.shape)
    # print( np.log(p))
    return -np.sum(y_h1 * np.log(p)) / y.shape[0]

# calculate the gradient of CE loss
def grad(imgs, y_hat, y):

    return np.dot(imgs, y_hat - y) / imgs.shape[1]

# create predictions based of of the given w
def predict(w, imgs):
    # pre-activation scores
    z = np.dot(imgs.T, w)
    # print(z)

    # convert to probabilities
    y_hat = np.exp(z) / np.sum(np.exp(z))

    return np.argmax(y_hat, axis=1)

# create predictions based of of the given w
def predict_prob(w, imgs):
    # pre-activation scores
    z = np.dot(imgs.T, w)
    # print(z)

    # convert to probabilities
    y_hat = np.exp(z) / np.sum(np.exp(z))

    return y_hat

# calculate the accuracy of the weights on the given data
def calc_accuracy(w, imgs, y):
    p = predict(w, imgs)

    r = p == y
    return np.mean(r)

# Given training and testing data, learning rate epsilon, and a specified batch size,
# conduct stochastic gradient descent (SGD) to optimize the weight matrix W (785x10).
# Then return W.
def softmaxRegression (w, trainingImages, trainingLabels, testingImages, testingLabels, classes, epsilon = None, batchSize = None, epoches = 1):
    # print("softmax shapes")
    # print(trainingImages.shape)
    # print(trainingLabels.shape)
    # print(testingImages.shape)
    # print(testingLabels.shape)
    # print()

    num_batches = trainingImages.shape[1] // batchSize

    # train the weights
    print(epoches)
    for j in range(epoches):
        for i in range(1, num_batches+1):
            # features and labels for current batch
            batch_x = trainingImages[:, (i-1)*batchSize : (i)*batchSize]
            batch_y = trainingLabels[(i-1)*batchSize : (i)*batchSize]

            # get predictions
            y_hat = predict(w, batch_x)

            # convert labels to one hot vectors
            y_hat_1h = one_hot(y_hat, classes)

            y_1h = one_hot(batch_y, classes)

            g = grad(batch_x, y_hat_1h, y_1h)

            # pre-activation scores
            # z = np.dot(batch_x.T, w)
            # # print(z)

            # # convert to probabilities
            # y_hat = np.exp(z) / np.sum(np.exp(z))

            # y_1h = one_hot(batch_y)

            # g = grad(batch_x, y_hat, y_1h)
            

            w = w - (epsilon * g)

            if(num_batches-i < 20 and j == 0):
                print("Training Loss {}/{}: ".format(i, num_batches), CE_loss(w, trainingImages, trainingLabels, classes))

    return w

if __name__ == "__main__":
    # Load data
    trainingImages = np.load("data\\training\\fashion_mnist_train_images.npy")
    trainingLabels = np.load("data\\training\\fashion_mnist_train_labels.npy")
    testingImages = np.load("data\\testing\\fashion_mnist_test_images.npy")
    testingLabels = np.load("data\\testing\\fashion_mnist_test_labels.npy")

    # squash pixel values between 0 and 1
    trainingImages = trainingImages / 255
    testingImages = testingImages / 255

    # print("initial shapes")
    # print(trainingImages.shape)
    # print(trainingLabels.shape)
    # print(testingImages.shape)
    # print(testingLabels.shape)
    # print()
    # print(np.amax(trainingImages))
    # print(np.amax(testingImages))

    # shuffle the data
    trainingLabels = np.array([trainingLabels])
    testingLabels = np.array([testingLabels])
    training_data = np.concatenate((trainingImages, trainingLabels.T), axis=1)
    testing_data = np.concatenate((testingImages, testingLabels.T), axis=1)

    np.random.shuffle(training_data)
    np.random.shuffle(testing_data)

    # reseparate the data
    trainingImages = training_data[:, :-1]
    trainingLabels = training_data[:, -1:].T[0]

    testingImages = testing_data[:, :-1]
    testingLabels = testing_data[:, -1:].T[0]

    # print("shuffle shapes")
    # print(trainingImages.shape)
    # print(trainingLabels.shape)
    # print(testingImages.shape)
    # print(testingLabels.shape)
    # print()

    # Append a constant 1 term to each example to correspond to the bias terms
    trainingImages = append1s(trainingImages)
    testingImages = append1s(testingImages)

    # print("reshape shapes")
    # print(trainingImages.shape)
    # print(trainingLabels.shape)
    # print(testingImages.shape)
    # print(testingLabels.shape)
    # print()

    W = np.random.randn(785, 10)
    W = softmaxRegression(W, trainingImages, trainingLabels, testingImages, testingLabels, 10, epsilon=0.1, batchSize=100, epoches=30)

    print("Training Loss: ", CE_loss(W, trainingImages, trainingLabels, 10))
    print("Training Accuracy: ", calc_accuracy(W, trainingImages, trainingLabels))
    print("Testing Loss: ", CE_loss(W, testingImages, testingLabels, 10))
    print("Testing Accuracy: ", calc_accuracy(W, testingImages, testingLabels))

    print("Predictions: ", predict(W, testingImages)[:10])
    print("Labels: ", testingLabels.astype(int)[:10])
    
    # Visualize the vectors
    titles = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    fig=plt.figure(figsize=(28, 28))
    for i in range(1, 11):
        fig.add_subplot(2, 5, i)
        plt.imshow(W[:-1, i-1].reshape((28, 28)).T, cmap='gray')
        plt.title(titles[i-1])
        
    plt.show()
