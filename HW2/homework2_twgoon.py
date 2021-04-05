import numpy as np
from numpy.core.numeric import ones
import matplotlib.pyplot as plt

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    samples = np.ones((x.shape[0], 1))
    
    # create feature that are powers of x
    for i in range(1, d+1):
        x_pow = x**i
        x_pow = np.reshape(x_pow, (-1, 1))
        print(x_pow.shape)
        samples = np.concatenate((samples, x_pow), axis=1)
    
    # calculate the optimal weights
    w = np.linalg.solve(
        np.dot(samples.T, samples),
        np.dot(samples.T, y)
    )

    return w

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):

    # Flatten images
    faces = np.reshape(faces, (-1, faces.shape[1]**2))

    # Append one onto each image
    ones = np.ones((faces.shape[0], 1))
    faces = np.concatenate((faces, ones), axis=1)

    faces = faces.T
    # print("append ",faces.shape)

    return faces

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    r = np.dot(Xtilde.T, w)
    a = np.mean((r - y)**2)/2
    return a

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    # compute the gradient
    g = np.dot(Xtilde,
        (np.dot(Xtilde.T, w)) - y)
    g = g / Xtilde.shape[1]

    # regularization
    w_no_bias = w[:w.shape[0]-1]
    w_no_bias = np.concatenate((w_no_bias, [0]))
    g = g + ((alpha*2*w_no_bias) / Xtilde.shape[1])

    return g

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):

    w = np.linalg.solve(
        np.dot(Xtilde, Xtilde.T),
        np.dot(Xtilde, y)
    )

    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):

    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations

    w = np.random.randn(2305)

    for i in range(T):
        # compute the gradient
        g = gradfMSE(w, Xtilde, y, alpha)

        # update the weights
        w = w - (EPSILON * g)
    
    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("data\\training\\age_regression_Xtr.npy"))
    ytr = np.load("data\\training\\age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("data\\testing\\age_regression_Xte.npy"))
    yte = np.load("data\\testing\\age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    # Report fMSE cost using each of the three learned weight vectors
    print("Training Loss Anlytical: ", fMSE(w1, Xtilde_tr, ytr))
    print("Testing Loss Anlytical: ", fMSE(w1, Xtilde_te, yte))

    print("Training Loss GD: ", fMSE(w2, Xtilde_tr, ytr))
    print("Testing Loss GD: ", fMSE(w2, Xtilde_te, yte))

    print("Training Loss reg: ", fMSE(w3, Xtilde_tr, ytr))
    print("Testing Loss reg: ", fMSE(w3, Xtilde_te, yte))

    with open("w.npy", "wb") as f:
        np.save(f, w1)
        np.save(f, w2)
        np.save(f, w3)

    im = np.reshape(w1[:w1.shape[0]-1], (48, 48))
    plt.imshow(im) 
    plt.title("Anlytical Method")
    plt.show()

    im = np.reshape(w2[:w2.shape[0]-1], (48, 48))
    plt.imshow(im) 
    plt.title("Gradient Descent")
    plt.show()

    im = np.reshape(w3[:w3.shape[0]-1], (48, 48))
    plt.imshow(im) 
    plt.title("Gradient Descent with Regularization")
    plt.show()
