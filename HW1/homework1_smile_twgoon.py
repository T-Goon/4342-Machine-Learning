import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    """Returns the percent similarity of 2 1d numpy arrays"""
    # compare vs labels
    compare = np.equal(y, np.transpose(yhat))

    # number true over total labels
    return np.count_nonzero(compare, axis=1) / y.shape[0]

def measureAccuracyOfPredictors (predictors, X, y):
    """Returns the accuracy of a set of predictors"""

    # array of pixel values predictors for each image
    pix1 = X[:, predictors[:, :, 0], predictors[:, :, 1]]
    pix2 = X[:, predictors[:, :, 2], predictors[:, :, 3]]

    # compare all of the predictor pixels
    results = pix1 > pix2
    del pix1
    del pix2

    # average all of the predictors
    results_avg = np.mean(results, axis=2)
    del results

    # convert to boolean
    results = np.greater(results_avg, np.full(results_avg.shape, .5))
    del results_avg

    return fPC(y, results)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, n):
    """Trains the classifier with stepwise regression"""
    batch_size = 100

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

        # measure accuracy on training data
        # do in batches to save memory
        batch_num = 0
        acc = list()
        while(batch_num < n):

            if(n - batch_num >= batch_size):
                
                acc.append(measureAccuracyOfPredictors(all_predictors, 
                trainingFaces[batch_num:(batch_num+batch_size)], 
                trainingLabels[batch_num:(batch_num+batch_size)]))

                batch_num += batch_size

            elif(n - batch_num < batch_size):
                acc.append(measureAccuracyOfPredictors(all_predictors, 
                trainingFaces[batch_num:], 
                trainingLabels[batch_num:]))

                batch_num = n

        # mean accuraccy from all batches to find total accuracy on all images
        acc = np.array(acc)
        acc = np.mean(acc, axis=0)

        # find index of best predictor
        max = np.argmax(acc)

        # save the predictor found
        predictors[i, 0] = all_predictors[max, i, 0]
        predictors[i, 1] = all_predictors[max, i, 1]
        predictors[i, 2] = all_predictors[max, i, 2]
        predictors[i, 3] = all_predictors[max, i, 3]

        print("N: {} Iteration: {} Training Accuracy: {}".format(n, i, acc[max]))

        # Print test accuracy
        # measure accuracy on testing data
        batch_num = 0
        test_acc = list()
        while(batch_num < testingFaces.shape[0]):

            if(testingFaces.shape[0] - batch_num >= batch_size):
                
                test_acc.append(measureAccuracyOfPredictors(np.array([all_predictors[max]]), 
                testingFaces[batch_num:(batch_num+batch_size)], 
                testingLabels[batch_num:(batch_num+batch_size)]))

                batch_num += batch_size

            elif(testingFaces.shape[0] - batch_num < batch_size):
                test_acc.append(measureAccuracyOfPredictors(np.array([all_predictors[max]]), 
                testingFaces[batch_num:], 
                testingLabels[batch_num:]))

                batch_num = testingFaces.shape[0]

        test_acc = np.array(test_acc)
        test_acc = np.mean(test_acc, axis=0)

        print("N: {} Iteration: {} Test Accuracy: {}".format(n, i, test_acc[0]))

        print("Predictors:\n", predictors)
        print()

    return predictors

def loadData (which):
    faces = np.load("data\\{}ing\\{}ingFaces.npy".format(which, which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("data\\{}ing\\{}ingLabels.npy".format(which, which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    n = [400, 800, 1200, 1600, 2000]
    p = None
    for i in n:
        p = stepwiseRegression(trainingFaces[:i], trainingLabels[:i], testingFaces, testingLabels, i)
        print("---------------------------------")

    show = True    
    if show:        # Show an arbitrary test image in grayscale        
        im = testingFaces[0,:,:]        
        fig,ax = plt.subplots(1)        
        ax.imshow(im, cmap='gray')        # Show r1,c1        
        for r1, c1, r2, c2 in p:
            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')        
            ax.add_patch(rect)        # Show r2,c2        
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')        
            ax.add_patch(rect)        # Display the merged result        

        plt.show()