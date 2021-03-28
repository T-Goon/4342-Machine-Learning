import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    """Returns the percent similarity of 2 1d numpy arrays"""
    return np.nonzero(np.equal(y, yhat))[0].shape[0] / y.shape[0]

def measureAccuracyOfPredictors (predictors, X, y):
    """Returns the accuracy of a set of predictors"""
    all_pred = np.array([])

    print(predictors)
    print(np.not_equal(predictors, np.full((5, 4), -1)))
    p = np.array([r for r in predictors if r[0] != -1])
    # array of pixel values predictors for each image
    pix1 = X[:, p[:, 0], p[:, 1]]
    pix2 = X[:, p[:, 2], p[:, 3]]

    results = pix1 > pix2

    print(results)

    # for image in X:
    #     pred = np.array([])
    #     for p in [r for r in predictors if r[0] != -1]: # for all valid predictors

    #         # compare 2 pixels
    #         if image[int(p[0]), int(p[1])] > image[int(p[2]), int(p[3])]:
    #             pred = np.append(pred, [1])
    #         else:
    #             pred = np.append(pred, [0])
        
    #     # Final verdict of all predictors
    #     if (np.nonzero(pred)[0].shape[0] / pred.shape[0]) > .5:
    #         all_pred = np.append(all_pred, [1])
    #     else:
    #         all_pred = np.append(all_pred, [0])

    return

    return fPC(all_pred, y)

def stepwiseRegression0 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    """Trains the classifier with stepwise regression"""
    show = False

    predictors = np.zeros((5, 4)) + -1

    for i in range(5): # 5 predictors
        max = float('-inf')

        for c1 in range(24): # coordinates of first pixel
            for r1 in range(24):

                for c2 in range(24): # coordinates of second pixel
                    for r2 in range(24):

                        p = predictors.copy()

                        p[i, 0] = c1
                        p[i, 1] = r1
                        p[i, 2] = c2
                        p[i, 3] = r2

                        # Assess new predictor for training
                        train_acc = measureAccuracyOfPredictors(p, trainingFaces, trainingLabels)

                        # Keep track of best predictors
                        if train_acc >= max:
                            max = train_acc

                            predictors[i, 0] = c1
                            predictors[i, 1] = r1
                            predictors[i, 2] = c2
                            predictors[i, 3] = r2

                        if show:
                            # Show an arbitrary test image in grayscale
                            im = testingFaces[0,:,:]
                            fig,ax = plt.subplots(1)
                            ax.imshow(im, cmap='gray')
                            # Show r1,c1
                            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            # Show r2,c2
                            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
                            ax.add_patch(rect)
                            # Display the merged result
                            plt.show()

                print("r1 ", r1)
            print("c1 ", c1)
        train_acc = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        print("Iteration: {} Training Accuracy: {}".format(i, train_acc))

        # Print test accuracy
        test_acc = measureAccuracyOfPredictors(p, trainingFaces, trainingLabels)
        print("Iteration: {} Test Accuracy: {}".format(i, test_acc))

        print("Predictors: ", predictors)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels, n):
    """Trains the classifier with stepwise regression"""
    show = False

    predictors = np.full((5, 4), 1)

    indexes = np.arange(0, 24)
    all_indexs = np.array(np.meshgrid(indexes, indexes, indexes, indexes)).T.reshape(-1, 4)

    print(all_indexs)

    for i in np.arange(0, 5):

        all_predictors = np.full((331776, 5, 4), -1)
        all_predictors[:, :i, :] = predictors[:i, :]

        print(all_predictors)

        all_predictors[:, i, :] = all_indexs
        
        print(all_predictors)

        acc = measureAccuracyOfPredictors(all_predictors, trainingFaces[:n], trainingLabels[:n])

        # print(acc)
        pass

    # for i in range(5): # 5 predictors
    #     max = float('-inf')

    #     for c1 in range(24): # coordinates of first pixel
    #         for r1 in range(24):

    #             for c2 in range(24): # coordinates of second pixel
    #                 for r2 in range(24):

    #                     p = predictors.copy()

    #                     p[i, 0] = c1
    #                     p[i, 1] = r1
    #                     p[i, 2] = c2
    #                     p[i, 3] = r2

    #                     # Assess new predictor for training
    #                     train_acc = measureAccuracyOfPredictors(p, trainingFaces, trainingLabels)

    #                     # Keep track of best predictors
    #                     if train_acc >= max:
    #                         max = train_acc

    #                         predictors[i, 0] = c1
    #                         predictors[i, 1] = r1
    #                         predictors[i, 2] = c2
    #                         predictors[i, 3] = r2

    #                     if show:
    #                         # Show an arbitrary test image in grayscale
    #                         im = testingFaces[0,:,:]
    #                         fig,ax = plt.subplots(1)
    #                         ax.imshow(im, cmap='gray')
    #                         # Show r1,c1
    #                         rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
    #                         ax.add_patch(rect)
    #                         # Show r2,c2
    #                         rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
    #                         ax.add_patch(rect)
    #                         # Display the merged result
    #                         plt.show()
                            
    #             print("r1 ", r1)
    #         print("c1 ", c1)
    #     train_acc = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
    #     print("Iteration: {} Training Accuracy: {}".format(i, train_acc))

    #     # Print test accuracy
    #     test_acc = measureAccuracyOfPredictors(p, testingFaces, testingLabels)
    #     print("Iteration: {} Test Accuracy: {}".format(i, test_acc))

    #     print("Predictors: ", predictors)

def loadData (which):
    faces = np.load("data\\{}ing\\{}ingFaces.npy".format(which, which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("data\\{}ing\\{}ingLabels.npy".format(which, which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")

    # n = entire training set
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, trainingFaces.shape[0])
