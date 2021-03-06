import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

def phiPoly3 (x):
    X_poly3 = np.array([
        np.ones(x.shape[0]),
        (3**(1/2))*x[:, 1],
        (3**(1/2))*x[:, 1]**2,
        x[:, 1]**3,
        (3**(1/2))*x[:, 0],
        (6**(1/2))*x[:, 0]*x[:, 1],
        (3**(1/2))*x[:, 0]*(x[:, 1]**2),
        (3**(1/2))*(x[:, 0]**2),
        (3**(1/2))*(x[:, 0]**2)*x[:, 1],
        (x[:, 0]**3)
    ])

    return X_poly3.T

def kerPoly3 (x, xprime):
    kp3 = np.zeros((x.shape[0], xprime.shape[0]))

    for i in range(x.shape[0]):
        for j in range(xprime.shape[0]):
            kp3[i, j] = (1+np.dot(x[i].T, xprime[j]))**3

    return kp3

    

def showPredictions (title, svm, x, denseCoords):  # feel free to add other parameters if desired
    #plt.scatter(..., ...)  # positive examples
    #plt.scatter(..., ...)  # negative examples

    # get predictions from the trained SVM
    pred = svm.predict(x)

    # Separate out the different predictions
    idxsNeg = np.nonzero(pred == -1)[0]
    idxsPos = np.nonzero(pred == 1)[0]

    # Plot
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([ "Lung disease", "No lung disease" ])
    plt.title(title)
    plt.show()

    pred_dense = svm.predict(denseCoords)

    # Separate out the different predictions
    idxsNeg_dense = np.nonzero(pred_dense == -1)[0]
    idxsPos_dense = np.nonzero(pred_dense == 1)[0]

    plt.scatter(all_coordinates[idxsPos_dense, 0], all_coordinates[idxsPos_dense, 1])
    plt.scatter(all_coordinates[idxsNeg_dense, 0], all_coordinates[idxsNeg_dense, 1])

    plt.xlabel("Radon")
    plt.ylabel("Asbestos")
    plt.legend([ "Lung disease", "No lung disease" ])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Load training data
    d = np.load("lung_toy.npy")
    X = d[:,0:2]  # features
    y = d[:,2]  # labels

    # Show scatter-plot of the data
    idxsNeg = np.nonzero(y == -1)[0]
    idxsPos = np.nonzero(y == 1)[0]
    plt.scatter(X[idxsPos, 0], X[idxsPos, 1])
    plt.scatter(X[idxsNeg, 0], X[idxsNeg, 1])
    plt.show()

    numbers = np.arange(0, 200, 2)
    numbers2 = np.arange(0, 11, .1)
    all_coordinates = np.array(np.meshgrid(numbers2, numbers)).T.reshape(-1, 2)

    # (a) Train linear SVM using sklearn
    svmLinear = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmLinear.fit(X, y)
    showPredictions("Linear", svmLinear, X, all_coordinates)

    # (b) Poly-3 using explicit transformation phiPoly3

    pp3 = phiPoly3(X)
    svmPoly = sklearn.svm.SVC(kernel='linear', C=0.01)
    svmPoly.fit(pp3, y)
    showPredictions("phiPoly3", svmPoly, pp3, phiPoly3(all_coordinates))
    
    # (c) Poly-3 using kernel matrix constructed by kernel function kerPoly3
    kp_train = kerPoly3(X, X)
    kp_test = kerPoly3(all_coordinates, X)

    svmKer = sklearn.svm.SVC(kernel='precomputed', C=0.01)
    svmKer.fit(kp_train, y)
    showPredictions("kerPoly3", svmKer, kp_train, kp_test)
    
    # (d) Poly-3 using sklearn's built-in polynomial kernel
    svmkernel = sklearn.svm.SVC(kernel='poly', C=0.01, gamma=1, coef0=1, degree=3)
    svmkernel.fit(X, y)
    showPredictions("kernel", svmkernel, X, all_coordinates)

    # (e) RBF using sklearn's built-in polynomial kernel
    svmRBF = sklearn.svm.SVC(kernel='rbf', C=1.0, gamma=1)
    svmRBF.fit(X, y)
    showPredictions("RBF1", svmRBF, X, all_coordinates)

    svmRBF = sklearn.svm.SVC(kernel='rbf', C=1.0, gamma=.03)
    svmRBF.fit(X, y)
    showPredictions("RBF2", svmRBF, X, all_coordinates)