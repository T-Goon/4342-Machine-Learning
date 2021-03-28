import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return (np.dot(A, B)) - C

def problem3 (A, B, C):
    return (A*B) + np.transpose(C)

def problem4 (x, S, y):
    return np.dot(np.dot(np.transpose(x), S), y)

def problem5 (A):
    return np.zeros(np.array(A).shape)

def problem6 (A):
    return np.ones(np.array(A).shape[0])

def problem7 (A, alpha):
    return A + (alpha * np.eye(A.shape[0]))

def problem8 (A, i, j):
    return A[i][j]

def problem9 (A, i):
    return np.sum(A[i, :])

def problem10 (A, c, d):
    return np.mean(A[np.nonzero(np.logical_and(A >= c, A <= d))])

def problem11 (A, k):
    w, v =  np.linalg.eig(A)
    w = np.argsort(w)
    w = np.flip(w)

    return v[:, w[:k]]

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A), np.transpose(x)))
