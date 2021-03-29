import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    """Returns the percent similarity of 2 1d numpy arrays"""
    # compare vs labels
    compare = np.equal(y, np.transpose(yhat))

    # number true over total labels
    return np.count_nonzero(compare, axis=1) / y.shape[0]

def comparePredictors(first, last, predictors, X):
    # array of pixel values predictors for each image
    pix1 = X[first:last, predictors[:, :, 0], predictors[:, :, 1]]
    pix2 = X[first:last, predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    return pix1 > pix2

def measureAccuracyOfPredictors (predictors, X, y):
    """Returns the accuracy of a set of predictors"""

    # all_comps = comparePredictors(0, (X.shape[0]//10), predictors, X)
    # for i in np.arange(1, 10):
    #     all_comps = np.concatenate((all_comps, 
    #     comparePredictors((X.shape[0]//10)*i, (X.shape[0]//10)*i+1, predictors, X)), axis=0)

    # all_comps = np.concatenate((all_comps, 
    # comparePredictors((X.shape[0]//10)*9, X.shape[0], predictors, X)), axis=0)

    # # array of pixel values predictors for each image
    # pix1 = X[:(X.shape[0]//10), predictors[:, :, 0], predictors[:, :, 1]]
    # pix2 = X[:(X.shape[0]//10), predictors[:, :, 2], predictors[:, :, 3]]

    # # compare all of the predictor pixels
    # all_comps = pix1 > pix2
    # del pix1
    # del pix2

    # # compare pixels in 10 steps to save memory space
    # for i in np.arange(1, 10):
    #     pix3 = X[((X.shape[0]//10)*i):((X.shape[0]//10)*i+1), predictors[:, :, 0], predictors[:, :, 1]]
    #     pix4 = X[((X.shape[0]//10)*i):((X.shape[0]//10)*i+1), predictors[:, :, 2], predictors[:, :, 3]]
    #     print(pix3.shape)

    #     # compare all of the predictor pixels
    #     results = pix3 > pix4
    #     del pix3
    #     del pix4

    #     all_comps = np.concatenate((all_comps, results), axis=0)
    #     del results
    # print(all_comps.shape)

    # array of pixel values predictors for each image
    # pix1 = X[((X.shape[0]//10)*10):, predictors[:, :, 0], predictors[:, :, 1]]
    # pix2 = X[((X.shape[0]//10)*10):, predictors[:, :, 2], predictors[:, :, 3]]

    # # compare all of the predictor pixels
    # all_comps = pix1 > pix2
    # del pix1
    # del pix2

    # array of pixel values predictors for each image
    pix1 = X[:(X.shape[0]//4), predictors[:, :, 0], predictors[:, :, 1]]
    pix2 = X[:(X.shape[0]//4), predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results1 = pix1 > pix2
    del pix1
    del pix2

    pix3 = X[(X.shape[0]//4):((X.shape[0]//4)*2), predictors[:, :, 0], predictors[:, :, 1]]
    pix4 = X[(X.shape[0]//4):((X.shape[0]//4)*2), predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results2 = pix3 > pix4
    del pix3
    del pix4

    pix5 = X[((X.shape[0]//4)*2):((X.shape[0]//4)*3), predictors[:, :, 0], predictors[:, :, 1]]
    pix6 = X[((X.shape[0]//4)*2):((X.shape[0]//4)*3), predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results3 = pix5 > pix6
    del pix5
    del pix6

    pix7 = X[((X.shape[0]//4)*3):((X.shape[0]//4)*4), predictors[:, :, 0], predictors[:, :, 1]]
    pix8 = X[((X.shape[0]//4)*3):((X.shape[0]//4)*4), predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results4 = pix7 > pix8
    del pix7
    del pix8

    all_comps = np.concatenate((results1, results2), axis=0)
    del results1
    del results2

    all_comps = np.concatenate((all_comps, results3), axis=0)
    del results3
    all_comps = np.concatenate((all_comps, results4), axis=0)
    del results4

    # average all of the predictors
    results_avg = np.mean(all_comps, axis=2)
    del all_comps

    # convert to boolean
    results = np.greater(results_avg, np.full(results_avg.shape, .5))
    del results_avg

    return fPC(y, results)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, n):
    """Trains the classifier with stepwise regression"""
    show = False

    predictors = np.full((5, 4), None)

    # all possible permutations of 4 numbers from 0 to 23
    indexes = np.arange(0, 24)
    all_indexs = np.array(np.meshgrid(indexes, indexes, indexes, indexes)).T.reshape(-1, 4)

    for i in range(5):

        all_predictors = np.full((331776, i+1, 4), -1)
        # copy over found predictor to all new predictors
        all_predictors[:, :i, :] = predictors[:i, :]

        # all possible new predictors
        all_predictors[:, i, :] = all_indexs

        # measure accuracy
        acc = measureAccuracyOfPredictors(all_predictors, trainingFaces[:n], trainingLabels[:n])

        # find index of best predictor
        max = np.argmax(acc)

        # save the predictor found
        predictors[i, 0] = all_predictors[max, i, 0]
        predictors[i, 1] = all_predictors[max, i, 1]
        predictors[i, 2] = all_predictors[max, i, 2]
        predictors[i, 3] = all_predictors[max, i, 3]

        print("N: {} Iteration: {} Training Accuracy: {}".format(n, i, acc[max]))

        # Print test accuracy
        test_acc = measureAccuracyOfPredictors(np.array([all_predictors[max]]), testingFaces, testingLabels)
        print("N: {} Iteration: {} Test Accuracy: {}".format(n, i, test_acc[0]))

        print("Predictors:\n", predictors)
        print()

def loadData (which):
    faces = np.load("data\\{}ing\\{}ingFaces.npy".format(which, which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("data\\{}ing\\{}ingLabels.npy".format(which, which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    # n = [400, 800, 1600, 2000]
    n = [2000]
    # n = entire training set
    for i in n:
        stepwiseRegression(trainingFaces[:i], trainingLabels[:i], testingFaces[:i], testingLabels[:i], i)
        print("---------------------------------")
