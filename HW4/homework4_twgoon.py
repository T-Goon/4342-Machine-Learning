from cvxopt import solvers, matrix
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import sklearn.svm

class SVM4342 ():
    def __init__ (self):
        pass

    # Expects each *row* to be an m-dimensional row vector. X should
    # contain n rows, where n is the number of examples.
    # y should correspondingly be an n-vector of labels (-1 or +1).
    def fit2 (self, X, y):
        # TODO change these -- they should be np.arrays representing matrices or vectors
        G = 0
        P = np.identity(X.shape[1]+1)
        q = np.zeros(X.shape[1]+1)
        h = 0

        print(X)
        # print(np.array([y]).T)
        y = np.array([y]).T
        # print(np.concatenate((X,y), axis=1))
        r = np.concatenate((X,y), axis=1)
        index = np.where(r[:, X.shape[1]] == 1)
        # print(index[0])

        c = r[index[0], :] * -1
        r[index[0], :] = c
        # print(r)

        X = r[:, :-1]
        y = r[:, -1]

        # print(X)
        # print(y)

        G = np.concatenate((X, np.array([np.ones(X.shape[0])]).T), axis=1)
        # h = np.array([y]).T
        h = np.ones(y.shape[0])
        # print(h)

        print(G)
        # print(matrix(G, tc='d'))
        # print(matrix(h, tc='d'))
        # print(matrix(P, tc='d'))
        # print(matrix(q, tc='d'))

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # print(np.array(sol['x']).T)
        w = np.array(sol['x']).T

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        
        self.w = w[0, :-1]
        self.b = w[0, -1]

    def fit (self, X, y):
        # TODO change these -- they should be np.arrays representing matrices or vectors
        G = 0
        P = np.identity(X.shape[1]+1)
        P[P.shape[0]-1, P.shape[1]-1] = 0
        q = np.zeros(X.shape[1]+1)
        h = 0

        X = -X
        print(y)

        # print(np.array([np.ones(X.shape[0])]).T)
        G = np.array([y]).T * np.concatenate((X, np.array([np.full(X.shape[0], -1)]).T), axis=1)
        # print(np.concatenate((X, np.array([np.ones(X.shape[0])]).T), axis=1))
        # print(G)

        h = np.full(y.shape[0], -1)
        # print(h)

        # Solve -- if the variables above are defined correctly, you can call this as-is:
        sol = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))

        # print("Solution ", np.array(sol['x']).T)
        w = np.array(sol['x']).T

        # Fetch the learned hyperplane and bias parameters out of sol['x']
        # To avoid any annoying errors due to broadcasting issues, I recommend
        # that you flatten() the w you retrieve from the solution vector so that
        # it becomes a 1-D np.array.
        
        self.w = w[0, :-1]
        self.b = w[0, -1]

    # Given a 2-D matrix of examples X, output a vector of predicted class labels
    def predict (self, x):
        r = np.dot(x, self.w) + self.b
        # print("r ", r)
        r[r > 0] = 1
        r[r < 0] = -1
        return r

def test1 ():
    # Set up toy problem
    X = np.array([ [1,1], [2,1], [1,2], [2,3], [1,4], [2,4] ])
    y = np.array([-1,-1,-1,1,1,1])

    # Train your model
    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard-margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

def test2 (seed):
    np.random.seed(seed)

    # Generate random data
    X = np.random.rand(20,3)
    # Generate random labels based on a random "ground-truth" hyperplane
    while True:
        w = np.random.rand(3)
        y = 2*(X.dot(w) > 0.5) - 1
        # Keep generating ground-truth hyperplanes until we find one
        # that results in 2 classes
        if len(np.unique(y)) > 1:
            break

    svm4342 = SVM4342()
    svm4342.fit(X, y)
    print(svm4342.w, svm4342.b)

    # Compare with sklearn
    svm = sklearn.svm.SVC(kernel='linear', C=1e15)  # 1e15 -- approximate hard margin
    svm.fit(X, y)
    print(svm.coef_, svm.intercept_)
    diff = np.linalg.norm(svm.coef_ - svm4342.w) + np.abs(svm.intercept_ - svm4342.b)
    print(diff)

    acc = np.mean(svm4342.predict(X) == svm.predict(X))
    print("Acc={}".format(acc))

    if acc == 1 and diff < 1e-1:
        print("Passed")

if __name__ == "__main__": 
    test1()

    for seed in range(5):
        test2(seed)
